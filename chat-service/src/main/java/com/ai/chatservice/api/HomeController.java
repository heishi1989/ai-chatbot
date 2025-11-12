package com.ai.chatservice.api;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

/** Trang chủ đơn giản để tránh 404 khi mở "/" */
@RestController
public class HomeController {
    @GetMapping("/")
    public String home() {
        return """
      Chat-Service đang chạy. Hãy gọi POST /api/chat/send với JSON:
      {"prompt":"Xin chào bot","maxNew":20}
      """;
    }
}
