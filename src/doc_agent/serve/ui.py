"""Stage 8 — Gradio demo"""
from __future__ import annotations
from ..contracts import *  # noqa

def launch(cfg: dict) -> None:
    """Gradio UI over the local pipeline."""
    try:
        import gradio as gr # type: ignore
    except ImportError as e:
        raise RuntimeError("Gradio is a critical dependency for the UI. Please install it with: pip install gradio") from e

    from .. import pipeline

    def ask_agent(query: str) -> str:
        if not query.strip():
            return "Please enter a question."
        
        try:
            ans = pipeline.answer(query, cfg)
            
            output = f"**Answer:** {ans.text}\n\n"
            output += f"---\n**Grounded:** {'✅ Yes' if ans.grounded else '❌ No'} | **Confidence:** {ans.confidence:.2f}\n"
            
            if ans.citations:
                output += "\n**Citations:**\n"
                for i, c in enumerate(ans.citations, 1):
                    output += f"- [{i}] `{c.chunk_id}` (Span: {c.span[0]}-{c.span[1]})\n"
            return output
        except Exception as e:
            return f"**Error:** {str(e)}"

    with gr.Blocks(title="Ar-Raheeq Al-Makhtum Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📖 Ar-Raheeq Al-Makhtum Assistant")
        gr.Markdown("Ask any question about the life of Prophet Muhammad (PBUH).")
        
        with gr.Row():
            with gr.Column(scale=4):
                query_input = gr.Textbox(
                    placeholder="e.g. হুদাইবিয়ার সন্ধির শর্তগুলো কী ছিল?", 
                    label="Your Question", 
                    lines=2
                )
            with gr.Column(scale=1):
                submit_btn = gr.Button("Ask Agent", variant="primary")
        
        answer_output = gr.Markdown(label="Response")
        
        submit_btn.click(fn=ask_agent, inputs=query_input, outputs=answer_output)
        query_input.submit(fn=ask_agent, inputs=query_input, outputs=answer_output)
        
        gr.Examples(
            examples=[
                "নবী মুহাম্মদ (সা.) কোন গোত্রে জন্মগ্রহণ করেন?",
                "হুদাইবিয়ার সন্ধির শর্তগুলো কী ছিল?",
                "বদর যুদ্ধে মুসলিম বাহিনীর সৈন্য সংখ্যা কত ছিল?"
            ],
            inputs=query_input
        )

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


