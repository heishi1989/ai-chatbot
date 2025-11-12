package com.ai.chatservice.core;

/**
 * Dàn nhạc: quyết định trả lời bằng Fallback (có nghĩa) hay bằng MiniGpt (học cơ chế).
 *  - Nếu Fallback nhận diện được ý định phổ biến → ưu tiên trả lời rõ ràng.
 *  - Nếu không → dùng MiniGptEngine sinh token (có thể "ngớ ngẩn" lúc đầu vì chưa có trọng số huấn luyện).
 * Sau này: khi bạn có tokenizer BPE + trọng số, chỉ cần tăng ưu tiên của MiniGptEngine là xong.
 */
public class ChatOrchestrator {

    public record Result(String text, String mode) {}

    private final SimpleTokenizer tok;
    private final MiniGptEngine gpt;
    private final ReplyFallback fb;

    public ChatOrchestrator(SimpleTokenizer tok, MiniGptEngine gpt, ReplyFallback fb){
        this.tok=tok; this.gpt=gpt; this.fb=fb;
    }

    public Result answer(String prompt, int maxNew, double topP, double temperature){
        // 1) Thử fallback trước để đảm bảo "có nghĩa"
        String ans = fb.tryAnswer(prompt);
        if (ans!=null) return new Result(ans, "fallback");

        // 2) Không khớp ý định → dùng LLM tự viết (để bạn quan sát pipeline)
        String gen = gpt.generate(tok, prompt, maxNew, topP, temperature);
        return new Result(gen, "mini-gpt");
    }
}
