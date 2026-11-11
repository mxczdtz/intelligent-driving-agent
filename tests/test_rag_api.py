"""RAG API 回归测试。所有 RAG 推理均使用 Mock，不下载模型。"""

from fastapi.testclient import TestClient

import app.main as main


def test_rag_ask_returns_answer_and_sources(monkeypatch):
    class FakeRAG:
        def answer(self, question: str, top_k: int):
            assert question == "发动机温度过高怎么办？"
            assert top_k == 2
            return {
                "answer": "应在安全地点停车，遵循手册。[KB001]",
                "sources": [
                    {
                        "id": "KB001",
                        "title": "发动机冷却液温度过高",
                        "source": "项目自编教学示例（非车厂维修手册）"
                    }
                ],
                "answer_mode": "extractive_high_risk",
            }

    monkeypatch.setattr(main, "get_rag_engine", lambda: FakeRAG())

    with TestClient(main.app) as client:
        response = client.post(
            "/rag/ask",
            json={"question": "发动机温度过高怎么办？", "top_k": 2}
        )

    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True
    assert data["sources"][0]["id"] == "KB001"
    assert data["answer_mode"] == "extractive_high_risk"
    assert "[KB001]" in data["answer"]


def test_rag_ask_rejects_blank_question(monkeypatch):
    # 输入校验应该在模型初始化之前完成。
    def forbidden():
        raise AssertionError("空问题不能调用 RAG 引擎")

    monkeypatch.setattr(main, "get_rag_engine", forbidden)

    with TestClient(main.app) as client:
        response = client.post(
            "/rag/ask",
            json={"question": "  ", "top_k": 3}
        )

    assert response.status_code == 422


def test_rag_ask_unavailable_returns_503(monkeypatch):
    def failed():
        raise RuntimeError("无法加载本地 Embedding 模型")

    monkeypatch.setattr(main, "get_rag_engine", failed)

    with TestClient(main.app) as client:
        response = client.post(
            "/rag/ask",
            json={"question": "制动故障怎么办？"}
        )

    assert response.status_code == 503
    assert "RAG 服务暂不可用" in response.json()["detail"]
    # 不将原始环境错误直接返回客户端
    assert "Embedding" not in response.json()["detail"]
