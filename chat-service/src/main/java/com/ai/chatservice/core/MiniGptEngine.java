package com.ai.chatservice.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Mô phỏng rất đơn giản 1 mô hình kiểu distilGPT-2:
 *  - Kiến trúc: token_emb + pos_emb → [N block: LN → Attention (multi-head) → residual → LN → MLP(GELU) → residual] → LN → lm_head
 *  - KV cache: lưu lại K,V của các token trước để suy luận token kế tiếp nhanh hơn (ở đây đơn giản hoá).
 *  - Trọng số random => CHỈ để minh hoạ cơ chế sinh token (không "hiểu nghĩa").
 *
 *  GỢI Ý: Ta dùng engine này khi fallback không tìm được ý định phổ biến.
 *  Muốn engine trả lời thật sự "có nghĩa", bạn sẽ cần:
 *    (1) tokenizer tốt (BPE), (2) trọng số đã huấn luyện, (3) hoặc tích hợp RAG.
 */
public class MiniGptEngine {

    // ===== Cấu hình mô hình =====
    private final int vocab, hidden, nLayers, nHeads, headDim, maxCtx;

    // Trọng số (đơn giản hoá dưới dạng mảng)
    private final float[][] tokenEmb; // [vocab][hidden]
    private final float[][] posEmb;   // [maxCtx][hidden]
    private final Block[] blocks;
    private final float[] lnFw, lnFb; // layernorm cuối
    private final float[][] lmHead;   // [hidden][vocab]

    private final Random rnd = new Random(42);

    public MiniGptEngine(int vocab, int hidden, int nLayers, int nHeads, int maxCtx){
        this.vocab=vocab; this.hidden=hidden; this.nLayers=nLayers; this.nHeads=nHeads; this.maxCtx=maxCtx;
        this.headDim = hidden / nHeads;

        // Khởi tạo trọng số random (minh hoạ)
        tokenEmb = new float[vocab][hidden];
        posEmb   = new float[maxCtx][hidden];
        rand2d(tokenEmb); rand2d(posEmb);

        blocks = new Block[nLayers];
        for (int i=0;i<nLayers;i++){
            blocks[i] = new Block(hidden, nHeads, headDim, maxCtx, rnd);
        }
        lnFw = ones(hidden); lnFb = zeros(hidden);
        lmHead = new float[hidden][vocab]; rand2d(lmHead);
    }

    // ======= API chính: sinh văn bản =======
    public String generate(SimpleTokenizer tok, String prompt, int maxNew, double topP, double temperature){
        int[] in = tok.encode(prompt);
        List<Integer> out = new ArrayList<>();
        int pos = 0, last = in.length>0 ? in[0] : 0;

        // Warm-up các token đầu vào
        for (int id : in){
            forwardSingle(id, pos);
            last = id; pos++;
            if (pos>=maxCtx) break;
        }

        // Sinh tiếp token mới
        for (int step=0; step<maxNew && pos<maxCtx; step++, pos++){
            float[] logits = forwardSingle(last, pos);
            int next = (topP<=0) ? SimpleSampler.greedy(logits) : SimpleSampler.topP(logits, topP, temperature);
            out.add(next);
            last = next;
        }
        return tok.decode(out);
    }

    // ======= forward 1 token (batch=1) =======
    private float[] forwardSingle(int tokenId, int pos){
        // 1) Lấy embedding token + embedding vị trí
        float[] x = new float[hidden];
        addVec(x, tokenEmb[tokenId]);
        addVec(x, posEmb[pos]);

        // 2) Qua N block Transformer
        for (Block b : blocks){
            x = b.forward(x, pos);
        }

        // 3) LayerNorm cuối
        layerNormInPlace(x, lnFw, lnFb, 1e-5f);

        // 4) Nhân linear ra logits vocab
        float[] logits = new float[vocab];
        for (int j=0;j<vocab;j++){
            float s=0f;
            for (int i=0;i<hidden;i++) s += x[i]*lmHead[i][j];
            logits[j] = s;
        }
        return logits;
    }

    // ======= Cấu phần bên trong =======
    private static class Block {
        final int hidden, nHeads, headDim, maxCtx;
        final float[] ln1w, ln1b, ln2w, ln2b;
        final float[][] Wq, Wk, Wv, Wo;  // [hidden][hidden]
        final float[] bq, bk, bv, bo;
        final float[][] W1, W2;          // MLP
        final float[] b1, b2;

        // KV cache
        final float[][] kCache; // [maxCtx][hidden]
        final float[][] vCache; // [maxCtx][hidden]

        Block(int hidden, int nHeads, int headDim, int maxCtx, Random rnd){
            this.hidden=hidden; this.nHeads=nHeads; this.headDim=headDim; this.maxCtx=maxCtx;
            ln1w = ones(hidden); ln1b = zeros(hidden);
            ln2w = ones(hidden); ln2b = zeros(hidden);
            Wq = rand2d(hidden, hidden, rnd); Wk = rand2d(hidden, hidden, rnd); Wv = rand2d(hidden, hidden, rnd); Wo = rand2d(hidden, hidden, rnd);
            bq = rand1d(hidden, rnd); bk = rand1d(hidden, rnd); bv = rand1d(hidden, rnd); bo = rand1d(hidden, rnd);
            W1 = rand2d(hidden, hidden*4, rnd); b1 = rand1d(hidden*4, rnd);
            W2 = rand2d(hidden*4, hidden, rnd); b2 = rand1d(hidden, rnd);
            kCache = new float[maxCtx][hidden];
            vCache = new float[maxCtx][hidden];
        }

        float[] forward(float[] xIn, int pos){
            // LN1
            float[] h1 = xIn.clone();
            layerNormInPlace(h1, ln1w, ln1b, 1e-5f);

            // Q,K,V
            float[] q = matmulAdd(h1, Wq, bq);
            float[] k = matmulAdd(h1, Wk, bk);
            float[] v = matmulAdd(h1, Wv, bv);

            // Lưu cache
            System.arraycopy(k, 0, kCache[pos], 0, hidden);
            System.arraycopy(v, 0, vCache[pos], 0, hidden);

            // Attention từng head
            float[] attnOut = new float[hidden];
            float invSqrt = (float)(1.0/Math.sqrt(headDim));
            for (int h=0; h<nHeads; h++){
                int base = h*headDim;
                // Tính điểm (scores) với tất cả token 0..pos
                float[] scores = new float[pos+1];
                for (int t=0;t<=pos;t++){
                    float s=0f;
                    for (int i=0;i<headDim;i++){
                        s += q[base+i] * kCache[t][base+i];
                    }
                    scores[t] = s * invSqrt;
                }
                softmaxInPlace(scores);
                // weighted sum V
                for (int i=0;i<headDim;i++){
                    float acc=0f;
                    for (int t=0;t<=pos;t++){
                        acc += scores[t] * vCache[t][base+i];
                    }
                    attnOut[base+i] = acc;
                }
            }
            float[] a = matmulAdd(attnOut, Wo, bo);

            // residual 1
            float[] y = add(xIn, a);

            // LN2
            float[] h2 = y.clone();
            layerNormInPlace(h2, ln2w, ln2b, 1e-5f);

            // MLP (GELU)
            float[] m1 = matmulAdd(h2, W1, b1);
            geluInPlace(m1);
            float[] m2 = matmulAdd(m1, W2, b2);

            // residual 2
            return add(y, m2);
        }
    }

    // ======= Toán học phụ trợ =======
    private static void geluInPlace(float[] x){
        for (int i=0;i<x.length;i++){
            float v=x[i];
            x[i] = 0.5f * v * (1f + (float)Math.tanh(Math.sqrt(2.0/Math.PI)*(v + 0.044715*Math.pow(v,3))));
        }
    }
    private static void layerNormInPlace(float[] x, float[] gamma, float[] beta, float eps){
        int D = x.length;
        float mean=0f; for(int i=0;i<D;i++) mean+=x[i]; mean/=D;
        float var=0f; for(int i=0;i<D;i++){ float d=x[i]-mean; var+=d*d; }
        float s = (float)Math.sqrt(var/D + eps);
        for (int i=0;i<D;i++){
            float norm = (x[i]-mean)/s;
            x[i] = norm*gamma[i] + beta[i];
        }
    }
    private static float[] matmulAdd(float[] x, float[][] W, float[] b){
        int IN = x.length, OUT = W[0].length;
        float[] y = new float[OUT];
        for (int j=0;j<OUT;j++){
            float sum=0f;
            for (int k=0;k<IN;k++) sum += x[k]*W[k][j];
            y[j] = sum + (b==null?0f:b[j]);
        }
        return y;
    }
    private static void softmaxInPlace(float[] a) {
        if (a == null || a.length == 0) return;

        // 1) Tìm max để ổn định số học (log-sum-exp trick)
        float max = a[0];
        for (int i = 1; i < a.length; i++) {
            if (a[i] > max) max = a[i];
        }

        // 2) e^(x - max) và tính tổng
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            a[i] = (float) Math.exp(a[i] - max);
            sum += a[i];
        }

        // 3) Chuẩn hoá về xác suất; phòng trường hợp sum=0 (rất hiếm)
        if (sum == 0.0) {
            float u = 1.0f / a.length;
            for (int i = 0; i < a.length; i++) a[i] = u;
            return;
        }
        float inv = (float) (1.0 / sum);
        for (int i = 0; i < a.length; i++) {
            a[i] *= inv;
        }
    }

    // ======= Tiện ích mảng =======
    private static void addVec(float[] dst, float[] src){ for (int i=0;i<dst.length;i++) dst[i]+=src[i]; }
    private static float[] add(float[] a, float[] b){ float[] r=a.clone(); for (int i=0;i<r.length;i++) r[i]+=b[i]; return r; }
    private static float[] ones(int n){ float[] a=new float[n]; for(int i=0;i<n;i++) a[i]=1f; return a; }
    private static float[] zeros(int n){ return new float[n]; }
    private static void rand2d(float[][] m){
        Random r = new Random(123);
        for(int i=0;i<m.length;i++) for(int j=0;j<m[i].length;j++) m[i][j]=(float)(r.nextGaussian()*0.02);
    }
    private static float[][] rand2d(int r, int c, Random rnd){
        float[][] m=new float[r][c];
        for(int i=0;i<r;i++) for(int j=0;j<c;j++) m[i][j]=(float)(rnd.nextGaussian()*0.02);
        return m;
    }
    private static float[] rand1d(int n, Random rnd){
        float[] a=new float[n]; for(int i=0;i<n;i++) a[i]=(float)(rnd.nextGaussian()*0.02); return a;
    }
}
