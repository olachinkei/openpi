#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import pathlib
import re
import sys
from typing import Iterable


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY", "wandb-smle")
DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "openpi-aloha-wandb-integration")


@dataclass(frozen=True)
class ReportTarget:
    key: str
    path: pathlib.Path
    default_title: str


@dataclass(frozen=True)
class ParsedBlock:
    kind: str
    text: str = ""
    level: int = 0


REPORT_TARGETS: dict[str, ReportTarget] = {
    "jp": ReportTarget(
        key="jp",
        path=SCRIPT_DIR / "wandb_report_jp.md",
        default_title="OpenPI x W&B: Physical AI ワークフローをつなぐ統合ガイド",
    ),
    "en": ReportTarget(
        key="en",
        path=SCRIPT_DIR / "wandb_report_en.md",
        default_title="OpenPI x W&B: Physical AI Workflow Integration Guide",
    ),
}


def _normalize_heading_text(text: str) -> str:
    cleaned = text.strip()
    if ":" in cleaned:
        prefix, suffix = cleaned.split(":", 1)
        if prefix.strip().lower() in {"title of report", "report title", "title"} and suffix.strip():
            return suffix.strip()
    return cleaned


def extract_report_title(markdown_text: str, default_title: str) -> str:
    for line in markdown_text.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", line.strip())
        if not match:
            continue
        candidate = _normalize_heading_text(match.group(1))
        lowered = candidate.lower()
        if lowered.startswith("this file is for developing"):
            continue
        if candidate:
            return candidate
    return default_title


def parse_markdown_sections(markdown_text: str, report_title: str) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    lines = markdown_text.splitlines()
    current_lines: list[str] = []
    in_code_block = False
    title_consumed = False

    def flush_markdown() -> None:
        content = "\n".join(current_lines).strip()
        current_lines.clear()
        if content:
            blocks.append(ParsedBlock(kind="markdown", text=content))

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            current_lines.append(line)
            in_code_block = not in_code_block
            continue

        if not in_code_block:
            heading_match = re.match(r"^(#{1,3})\s+(.+?)\s*$", stripped)
            if heading_match:
                flush_markdown()
                level = len(heading_match.group(1))
                heading_text = _normalize_heading_text(heading_match.group(2))

                if level == 1 and not title_consumed and heading_text == report_title:
                    title_consumed = True
                    continue

                title_consumed = title_consumed or level == 1
                blocks.append(ParsedBlock(kind="heading", text=heading_text, level=level))
                continue

            if stripped.lower() == "[toc]":
                flush_markdown()
                blocks.append(ParsedBlock(kind="toc"))
                continue

        current_lines.append(line)

    flush_markdown()
    return blocks


def import_report_api():
    try:
        import wandb.apis.reports.v2 as wr  # type: ignore

        if all(hasattr(wr, name) for name in ("Report", "MarkdownBlock", "H1", "TableOfContents")):
            return wr
    except ImportError:
        pass

    try:
        import wandb_workspaces.reports.v2 as wr  # type: ignore

        return wr
    except ImportError as exc:
        raise ImportError(
            "W&B Report API could not be imported. Install `wandb[workspaces]` or "
            "`wandb_workspaces` in the environment that runs this script."
        ) from exc


def build_wandb_blocks(parsed_blocks: Iterable[ParsedBlock]):
    wr = import_report_api()
    blocks = []
    for block in parsed_blocks:
        if block.kind == "heading":
            heading_factory = {1: wr.H1, 2: wr.H2, 3: wr.H3}[block.level]
            blocks.append(heading_factory(block.text))
        elif block.kind == "toc":
            blocks.append(wr.TableOfContents())
        elif block.kind == "markdown":
            blocks.append(wr.MarkdownBlock(block.text))
        else:
            raise ValueError(f"Unsupported block kind: {block.kind}")
    return blocks


def _prepare_report_content(target: ReportTarget) -> tuple[str, list[ParsedBlock]]:
    markdown_text = target.path.read_text(encoding="utf-8")
    report_title = extract_report_title(markdown_text, target.default_title)
    parsed_blocks = parse_markdown_sections(markdown_text, report_title)
    return report_title, parsed_blocks


def _maybe_login() -> None:
    import wandb

    api_key = os.environ.get("WANDB_API_KEY")
    if api_key and len(api_key.strip()) == 40:
        wandb.login(key=api_key.strip(), relogin=True)


def create_report(target: ReportTarget, *, entity: str, project: str, draft: bool) -> str:
    report_title, parsed_blocks = _prepare_report_content(target)

    _maybe_login()
    wr = import_report_api()
    report = wr.Report(
        entity=entity,
        project=project,
        title=report_title,
        description="",
        width="fluid",
    )
    report.blocks = build_wandb_blocks(parsed_blocks)
    report.save(draft=draft)
    return report.url


def update_report(target: ReportTarget, *, report_url: str, draft: bool) -> str:
    report_title, parsed_blocks = _prepare_report_content(target)

    _maybe_login()
    wr = import_report_api()
    report = wr.Report.from_url(report_url)
    report.title = report_title
    report.blocks = build_wandb_blocks(parsed_blocks)
    report.save(draft=draft)
    return report.url


def selected_targets(language: str) -> list[ReportTarget]:
    if language == "all":
        return [REPORT_TARGETS["jp"], REPORT_TARGETS["en"]]
    return [REPORT_TARGETS[language]]


def run_dry(targets: Iterable[ReportTarget]) -> int:
    for target in targets:
        markdown_text = target.path.read_text(encoding="utf-8")
        report_title = extract_report_title(markdown_text, target.default_title)
        parsed_blocks = parse_markdown_sections(markdown_text, report_title)

        heading_count = sum(1 for block in parsed_blocks if block.kind == "heading")
        markdown_block_count = sum(1 for block in parsed_blocks if block.kind == "markdown")
        toc_count = sum(1 for block in parsed_blocks if block.kind == "toc")

        print(f"[dry-run] {target.key}")
        print(f"  file: {target.path}")
        print(f"  title: {report_title}")
        print(f"  headings: {heading_count}")
        print(f"  markdown_blocks: {markdown_block_count}")
        print(f"  toc_blocks: {toc_count}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create W&B Reports from local markdown files. "
            "Each section body is uploaded as a MarkdownBlock."
        )
    )
    parser.add_argument(
        "--lang",
        choices=("jp", "en", "all"),
        default="all",
        help="Which markdown report to upload.",
    )
    parser.add_argument(
        "--entity",
        default=DEFAULT_ENTITY,
        help=f"W&B entity. Defaults to WANDB_ENTITY or {DEFAULT_ENTITY}.",
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help=f"W&B project. Defaults to WANDB_PROJECT or {DEFAULT_PROJECT}.",
    )
    parser.add_argument(
        "--draft",
        action="store_true",
        help="Create the report as a draft.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse markdown and print the planned blocks without uploading.",
    )
    parser.add_argument(
        "--report-url",
        help=(
            "Update an existing W&B report instead of creating a new one. "
            "Use only with a single language target."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    targets = selected_targets(args.lang)

    missing = [target.path for target in targets if not target.path.exists()]
    if missing:
        for path in missing:
            print(f"Missing markdown file: {path}", file=sys.stderr)
        return 1

    if args.dry_run:
        return run_dry(targets)

    if args.report_url and len(targets) != 1:
        print("--report-url can only be used with --lang jp or --lang en", file=sys.stderr)
        return 1

    urls: dict[str, str] = {}
    for target in targets:
        if args.report_url:
            url = update_report(
                target,
                report_url=args.report_url,
                draft=args.draft,
            )
        else:
            url = create_report(
                target,
                entity=args.entity,
                project=args.project,
                draft=args.draft,
            )
        urls[target.key] = url
        print(f"{target.key}: {url}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
