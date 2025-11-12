package com.ai.chatservice.core;

import java.util.*;

/**
 * Tokenizer "bình dân": tách theo khoảng trắng, ánh xạ từ→id.
 * => Dùng để minh hoạ pipeline LLM. Sau này thay bằng BPE để tốt hơn.
 */
public class SimpleTokenizer {
    private final Map<String,Integer> toId = new HashMap<>();
    private final Map<Integer,String> toWord = new HashMap<>();

    public void add(String w, int id){ toId.put(w, id); toWord.put(id, w); }
    public int size(){ return toId.size(); }

    public int[] encode(String text){
        String[] ws = text.trim().toLowerCase().split("\\s+");
        int[] ids = new int[ws.length];
        for (int i=0;i<ws.length;i++){
            ids[i] = toId.getOrDefault(ws[i], toId.getOrDefault("<unk>", 0));
        }
        return ids;
    }

    public String decode(List<Integer> ids){
        StringBuilder sb = new StringBuilder();
        for (int id: ids) sb.append(toWord.getOrDefault(id, "<unk>")).append(' ');
        return sb.toString().trim();
    }
}
