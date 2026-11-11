# GitHub 发布前检查清单

- [ ] `app/main.py`、`app/intent_router.py`、`app/rag_engine.py`、`app/qwen_engine.py`、`app/pending_actions.py` 都在
- [ ] `.gitignore` 已生效，`.env`、`models/`、`rag_data/`、`venv/` 均未暂存
- [ ] 没有把个人用户名、Windows 绝对路径、访问令牌、密钥、浏览器 Cookie 写进 README/源码
- [ ] 下载、替换并检查 `app/qwen_engine.py` 的 `QWEN_MODEL_DIR` 环境变量配置
- [ ] 在同一个 PowerShell 中设置 `QWEN_MODEL_DIR` 和 `RAG_EMBED_MODEL`
- [ ] `python -m pytest tests -v` 通过
- [ ] `python -m evaluation.run_rag_eval --mode both` 完成
- [ ] `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000` 启动成功
- [ ] Swagger 的 `/assistant/chat` 真实调用测试通过
- [ ] 说明本地教学模拟，不连接真实车辆；不声称生产级安全
- [ ] 检查 `git diff --cached --name-only`，再决定推送
