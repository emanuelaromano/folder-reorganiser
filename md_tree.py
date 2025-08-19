import os
import sys
import argparse
from typing import List, Iterable, Tuple, Optional, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
import json
load_dotenv()

DEFAULT_EXCLUDES = {
    ".DS_Store",
    "__pycache__",
    "node_modules",
    ".git",
    ".idea",
    ".venv",
    "venv",
}

INDENT = "    " 


def human_size(num: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if num < 1024.0:
            return f"{num:.0f}{unit}" if unit == "B" else f"{num:.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}PB"

def escape_md(name: str) -> str:
    for ch in ["\\", "*", "_", "`", "[", "]", "(", ")", "#", "+", "!", ">", "|"]:
        name = name.replace(ch, f"\\{ch}")
    return name

def iter_entries(path: str) -> Iterable[os.DirEntry]:
    try:
        with os.scandir(path) as it:
            entries = list(it)
    except PermissionError:
        return []
    entries.sort(key=lambda e: (not e.is_dir(follow_symlinks=False), e.name.lower()))
    return entries

def build_tree_json(
    root: str,
    max_depth: int = 0,
    include_hidden: bool = False,
    show_sizes: bool = False,
    excludes: set = DEFAULT_EXCLUDES,
    follow_symlinks: bool = False,
) -> Dict[str, Any]:
    root = os.path.abspath(os.path.expanduser(root))
    root_name = os.path.basename(root) or root

    seen_inodes: set[Tuple[int, int]] = set()  # for safe symlink following

    def should_skip(name: str) -> bool:
        if name in excludes: return True
        if not include_hidden and name.startswith("."): return True
        return False

    def inode_of(path: str) -> Tuple[int, int]:
        try:
            st = os.stat(path, follow_symlinks=follow_symlinks)
            return (st.st_dev, st.st_ino)
        except Exception:
            return (-1, -1)

    def make_node(path: str, relpath: str, depth: int) -> Dict[str, Any]:
        try:
            is_dir = os.path.isdir(path) if follow_symlinks else os.path.isdir(path) and not os.path.islink(path)
        except Exception:
            is_dir = False

        if is_dir:
            node: Dict[str, Any] = {
                "type": "dir",
                "name": os.path.basename(path) or relpath,
                "relpath": relpath or ".",
                "children": []
            }
            if max_depth and depth > max_depth:
                return node

            # handle symlink cycles if following
            if follow_symlinks:
                ino = inode_of(path)
                if ino in seen_inodes:
                    node["cycle_detected"] = True
                    return node
                seen_inodes.add(ino)

            for entry in iter_entries(path):
                if should_skip(entry.name):
                    continue
                child_rel = os.path.join(relpath, entry.name) if relpath else entry.name
                child_path = os.path.join(path, entry.name)

                try:
                    is_link = os.path.islink(child_path)
                    is_dir_child = entry.is_dir(follow_symlinks=follow_symlinks)
                except PermissionError:
                    node["children"].append({
                        "type": "unknown",
                        "name": entry.name,
                        "relpath": child_rel,
                        "error": "permission_denied"
                    })
                    continue

                if is_dir_child:
                    node["children"].append(make_node(child_path, child_rel, depth + 1))
                else:
                    file_node: Dict[str, Any] = {
                        "type": "file",
                        "name": entry.name,
                        "relpath": child_rel,
                        "is_link": is_link,
                        "ext": os.path.splitext(entry.name)[1]
                    }
                    if show_sizes:
                        try:
                            size_bytes = os.path.getsize(child_path)
                            file_node["size"] = human_size(size_bytes)
                            file_node["size_bytes"] = size_bytes
                        except Exception:
                            file_node["size"] = "?"
                    node["children"].append(file_node)

            return node
        else:
            # root could be a file
            node: Dict[str, Any] = {
                "type": "file",
                "name": os.path.basename(path),
                "relpath": relpath or os.path.basename(path),
                "is_link": os.path.islink(path),
                "ext": os.path.splitext(path)[1]
            }
            if show_sizes:
                try:
                    size_bytes = os.path.getsize(path)
                    node["size"] = human_size(size_bytes)
                    node["size_bytes"] = size_bytes
                except Exception:
                    node["size"] = "?"
            return node

    return make_node(root, relpath="", depth=0)


SYSTEM_PROMPT = (
    "You are an information architect and naming-conventions expert. "
    "Given a directory tree in JSON, produce:\n"
    "1) Unclear or inconsistent file/folder names with concrete rename suggestions.\n"
    "2) Reorganization suggestions (grouping, archival, deduplication, top-level structure). "
    "Be concise, actionable, and justify each suggestion briefly."
)

USER_PROMPT_TEMPLATE = """Analyze the following directory tree (JSON) and respond with two sections.

## Unclear File/Folder Namings
- Identify ambiguous, generic, duplicated, or noisy names (e.g., 'final', 'copy', 'v1', 'untitled', inconsistent casing/spaces).
- For each, propose a clearer name in `old → new` format and add a one-sentence rationale.
- Reference items by `relpath`.

## Reorganization Suggestions
- Propose a logical top-level structure (categories).
- Suggest merges/splits, archives for stale/unused areas, and consolidation of likely duplicates (names only).
- If sizes are present, call out unusually large folders; suggest where a README.md would help.

### Tree (JSON)
```json
{tree_json}
````

"""

def chunk_text(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            nl = text.rfind("\n", start, end)
            if nl > start + max_chars * 0.6:
                end = nl + 1
        chunks.append(text[start:end])
        start = end
    return chunks

def call_openai_on_tree_json(
    tree_json_str: str,
    model: str = "gpt-4o-mini",
    max_chars: int = 12000,
    api_key_env: str = "OPENAI_API_KEY",
    base_url: Optional[str] = None,
) -> str:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set")

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    chunks = chunk_text(tree_json_str, max_chars=max_chars)
    outputs: List[str] = []

    for idx, chunk in enumerate(chunks, 1):
        user_prompt = USER_PROMPT_TEMPLATE.format(tree_json=chunk)
        print("[LLM] Calling openai")
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        try:
            outputs.append(resp.output_text)
        except Exception:
            # Fallback parse for defensive coding
            choice_texts = []
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", "") == "message":
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", "") in ("output_text", "text"):
                            choice_texts.append(getattr(c, "text", ""))
            outputs.append("\n".join(choice_texts) if choice_texts else str(resp))

        if len(chunks) > 1 and idx < len(chunks):
            outputs.append(f"\n---\n_Chunk {idx}/{len(chunks)} end._\n")

    return "\n".join(outputs).strip() + "\n"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate a JSON directory tree and get LLM suggestions.')
    parser.add_argument('--path', type=str, default=os.path.join(os.path.expanduser("~"), "Desktop/video_interview/"),
                        help='The root path to start building the directory tree.')
    parser.add_argument('--json_out', type=str, default='tree.json',
                        help='The output file for the JSON directory tree.')
    parser.add_argument('--response_out', type=str, default='llm_suggestions.md',
                        help='The output file for the LLM suggestions.')
    parser.add_argument('--pretty', action='store_true',
                        help='Pretty print the JSON output.')
    parser.add_argument('--max_depth', type=int, default=0,
                        help='The maximum depth to traverse the directory tree.')
    parser.add_argument('--include_hidden', action='store_true',
                        help='Include hidden files and directories.')
    parser.add_argument('--show_sizes', action='store_true',
                        help='Show file sizes in the JSON output.')
    parser.add_argument('--follow_symlinks', action='store_true',
                        help='Follow symbolic links.')
    parser.add_argument('--no_default_excludes', action='store_true',
                        help='Do not exclude default directories like .git, node_modules, etc.')
    parser.add_argument('--exclude', type=str, nargs='*', default=[],
                        help='Additional directories or files to exclude.')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='The model to use for LLM suggestions.')
    parser.add_argument('--max_chars', type=int, default=12000,
                        help='The maximum number of characters per chunk for LLM input.')
    parser.add_argument('--api_key_env', type=str, default='OPENAI_API_KEY',
                        help='The environment variable for the OpenAI API key.')
    parser.add_argument('--base_url', type=str, default=None,
                        help='The base URL for the OpenAI API.')
    return parser.parse_args()


def main():
    args = parse_arguments()

    excludes = set()
    if not args.no_default_excludes:
        excludes |= set(DEFAULT_EXCLUDES)
    excludes |= set(args.exclude or [])

    tree = build_tree_json(
        root=args.path,
        max_depth=args.max_depth,
        include_hidden=args.include_hidden,
        show_sizes=args.show_sizes,
        excludes=excludes,
        follow_symlinks=args.follow_symlinks,
    )

    # Save JSON
    if args.json_out:
        try:
            with open(args.json_out, "w", encoding="utf-8") as f:
                if args.pretty:
                    json.dump(tree, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(tree, f, separators=(",", ":"), ensure_ascii=False)
            print(f"Wrote JSON tree to: {args.json_out}")
        except OSError as e:
            print(f"Error writing {args.json_out}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(json.dumps(tree, indent=2, ensure_ascii=False))

    # Prepare compact JSON for LLM (to save tokens)
    tree_json_str = json.dumps(tree, ensure_ascii=False, separators=(",", ":"))

    # Call LLM
    try:
        response_text = call_openai_on_tree_json(
            tree_json_str=tree_json_str,
            model=args.model,
            max_chars=args.max_chars,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
        )
    except Exception as e:
        print(f"[LLM] Error: {e}", file=sys.stderr)
        sys.exit(2)

    # Save LLM response
    try:
        with open(args.response_out, "w", encoding="utf-8") as f:
            f.write(response_text)
        print(f"Wrote LLM suggestions to: {args.response_out}")
    except OSError as e:
        print(f"Error writing {args.response_out}: {e}", file=sys.stderr)
        sys.exit(3)

if __name__ == "__main__":
    main()
