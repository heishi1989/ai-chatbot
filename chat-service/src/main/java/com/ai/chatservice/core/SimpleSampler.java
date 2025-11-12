package com.ai.chatservice.core;

import java.util.Random;

/**
 * Chọn token kế tiếp từ logits:
 *  - greedy: lấy id có xác suất cao nhất
 *  - top-p: chọn ngẫu nhiên trong nhóm có xác suất cộng dồn ≥ p
 */
public class SimpleSampler {
    private static final Random R = new Random();

    public static int greedy(float[] logits){
        int idx=0; float best = logits[0];
        for(int i=1;i<logits.length;i++){
            if (logits[i]>best){ best=logits[i]; idx=i; }
        }
        return idx;
    }

    public static int topP(float[] logits, double topP, double temperature){
        int V = logits.length;
        // softmax( logits / temperature )
        float max = logits[0]; for(int i=1;i<V;i++) max = Math.max(max, logits[i]);
        double[] p = new double[V]; double sum=0;
        for(int i=0;i<V;i++){
            double z = (logits[i]-max) / Math.max(1e-8, temperature);
            p[i] = Math.exp(z); sum += p[i];
        }
        for(int i=0;i<V;i++) p[i] /= sum;

        // sắp xếp giảm dần
        int[] idx = new int[V]; for(int i=0;i<V;i++) idx[i]=i;
        for(int i=1;i<V;i++){ int j=i; while(j>0 && p[idx[j-1]]<p[idx[j]]){ int t=idx[j]; idx[j]=idx[j-1]; idx[j-1]=t; j--; } }

        // cắt theo topP
        double csum=0; int cut=0;
        for(;cut<V;cut++){ csum+=p[idx[cut]]; if (csum>=topP){cut++; break;} }

        // bốc thăm trong "nhóm hợp lý"
        double r = R.nextDouble()*csum, acc=0;
        for(int i=0;i<cut;i++){ acc+=p[idx[i]]; if (r<=acc) return idx[i]; }
        return idx[0];
    }
}
