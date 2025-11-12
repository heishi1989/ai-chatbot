package com.ai.chatservice;

import com.ai.chatservice.core.ChatOrchestrator;
import com.ai.chatservice.core.MiniGptEngine;
import com.ai.chatservice.core.ReplyFallback;
import com.ai.chatservice.core.SimpleTokenizer;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

/**
 * Ứng dụng Spring Boot khởi động duy nhất.
 * Tại đây ta khởi tạo các "thành phần" lõi: tokenizer, engine LLM tự viết, fallback, orchestrator.
 */
@SpringBootApplication
public class ChatServiceApplication {

	public static void main(String[] args) {
		SpringApplication.run(ChatServiceApplication.class, args);
	}

    // Bean tokenizer cực đơn giản (từ vựng toy). Sau này thay bằng BPE chuẩn.
    @Bean
    public SimpleTokenizer tokenizer() {
        SimpleTokenizer t = new SimpleTokenizer();
        t.add("<unk>", 0);
        t.add("xin", 1);
        t.add("chào", 2);
        t.add("bạn", 3);
        t.add("tôi", 4);
        t.add("là", 5);
        t.add("trợ", 6);
        t.add("lý", 7);
        t.add("ảo", 8);
        t.add("rất", 9);
        t.add("vui", 10);
        t.add("được", 11);
        t.add("hỗ", 12);
        t.add("trợ", 13); // cố tình trùng “trợ” (ví dụ), tokenizer toy nên không hoàn hảo
        return t;
    }

    // Bean engine LLM tự viết (distilGPT-2 giản lược): trọng số random → học cơ chế sinh token
    @Bean
    public MiniGptEngine miniGptEngine(SimpleTokenizer tok) {
        // Cấu hình nho nhỏ: nhanh, dễ chạy
        int vocab = tok.size();
        return new MiniGptEngine(
                vocab,
                /*hidden*/ 128,
                /*nLayers*/ 3,
                /*nHeads*/ 8,
                /*maxCtx*/ 128
        );
    }

    // Bean fallback trả lời có nghĩa: nhận diện ý định cơ bản → trả câu tự nhiên tiếng Việt
    @Bean
    public ReplyFallback fallback() {
        return new ReplyFallback();
    }

    // Bean dàn nhạc: ưu tiên fallback (đảm bảo "có nghĩa"), nếu không trúng → dùng LLM tự viết
    @Bean
    public ChatOrchestrator orchestrator(SimpleTokenizer tok, MiniGptEngine gpt, ReplyFallback fb) {
        return new ChatOrchestrator(tok, gpt, fb);
    }

}
