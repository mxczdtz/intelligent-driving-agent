"""安全地为已有 app/main.py 增加路由，自动备份、可重复执行。"""
from datetime import datetime
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "app" / "main.py"
MARK = "# === SCENE_RISK_ROUTER_PHASE1 ==="
ADDITION = f"""
{MARK}
from app.scene_api import router as scene_risk_router
app.include_router(scene_risk_router)
"""


def main():
    if not TARGET.is_file():
        raise SystemExit(f"未找到旧项目主文件：{TARGET}")
    original = TARGET.read_text(encoding="utf-8-sig")
    if MARK in original:
        print("视觉路由已接入，无需重复修改。")
        return
    tree = ast.parse(original, filename=str(TARGET))
    if not any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and ((isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "app" for t in node.targets))
             or (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "app"))
        for node in tree.body
    ):
        raise SystemExit("未在 app/main.py 中确认 app 对象；没有修改文件。")
    backup_dir = ROOT / "evidence"
    backup_dir.mkdir(parents=True, exist_ok=True)  # 原项目 .gitignore 已忽略 evidence/
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = backup_dir / f"main_before_scene_{stamp}.py"
    backup.write_text(original, encoding="utf-8")
    modified = original.rstrip() + "\n" + ADDITION
    ast.parse(modified, filename=str(TARGET))
    TARGET.write_text(modified, encoding="utf-8")
    print("已接入 /scene/analyze")
    print(f"原主文件备份：{backup}")
    print("请运行 pytest 并检查 git diff，再决定是否提交。")


if __name__ == "__main__":
    main()
