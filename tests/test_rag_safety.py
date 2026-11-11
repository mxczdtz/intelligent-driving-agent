from app.rag_engine import VehicleRAG, load_knowledge


def test_high_risk_answer_uses_source_text_without_qwen(monkeypatch):
    """高风险故障不允许小模型把冷车检查误写成高温检查。"""
    documents = load_knowledge()
    kb001 = next(doc for doc in documents if doc["id"] == "KB001")

    rag = VehicleRAG.__new__(VehicleRAG)
    monkeypatch.setattr(rag, "search", lambda question, top_k=3: [kb001])

    result = rag.answer("发动机温度过高怎么办？")

    assert result["answer_mode"] == "extractive_high_risk"
    assert kb001["content"] in result["answer"]
    assert "[KB001]" in result["answer"]
    assert result["sources"][0]["id"] == "KB001"
    assert "待车辆充分冷却后" in result["answer"]
    assert "应在冷却系统高温时进行检查和处理" not in result["answer"]
