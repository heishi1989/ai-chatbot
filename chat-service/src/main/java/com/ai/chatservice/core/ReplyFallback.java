package com.ai.chatservice.core;

import java.time.LocalTime;

/**
 * Fallback “thông minh đơn giản”:
 * - Nhận diện một số ý định phổ biến (chào hỏi, giới thiệu, giờ giấc, trợ giúp).
 * - Trả lời ngắn gọn, tự nhiên tiếng Việt.
 * => Đảm bảo người dùng thấy câu trả lời "có nghĩa" ngay từ ngày đầu,
 *    trong khi engine LLM tự viết vẫn tồn tại để bạn học & sẽ được nâng cấp dần.
 */
public class ReplyFallback {

    public String tryAnswer(String user){
        if (user==null || user.isBlank()) return null;
        String s = user.trim().toLowerCase();

        // Chào hỏi
        if (s.matches(".*\\b(xin chào|chào|hello|hi|chao)\\b.*")) {
            return "Xin chào! Mình là trợ lý ảo. Mình có thể giúp gì cho bạn hôm nay?";
        }
        // Hỏi tên
        if (s.matches(".*\\b(bạn tên gì|tên bạn là gì|cậu tên gì)\\b.*")) {
            return "Mình là một trợ lý A.I mini. Bạn có thể gọi mình là Bot.";
        }
        // Hỏi giờ
        if (s.matches(".*\\b(mấy giờ|bây giờ là mấy giờ|giờ bao nhiêu)\\b.*")) {
            LocalTime now = LocalTime.now();
            return "Bây giờ là " + now.getHour() + " giờ " + now.getMinute() + " phút (theo máy chủ).";
        }
        // Cần giúp đỡ
        if (s.matches(".*\\b(giúp|hỗ trợ|support)\\b.*")) {
            return "Bạn cứ mô tả vấn đề/nghiệp vụ, mình sẽ gợi ý hướng giải quyết ngắn gọn nhé.";
        }
        // Tạm thời: không khớp ý định → trả null để Orchestrator dùng LLM
        return null;
    }
}
