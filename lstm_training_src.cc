/** \file lstm.h
 *  * Contains a simple C interface to call lstm training
 *    */

#pragma once

#include "mkl.h"
#include "math.h"
#ifdef __cplusplus
#include <cstddef>
extern "C" {
#endif
#define max(a,b) ((a) > (b) ? (a) : (b))

void  LSTM_batch_gemm(int batch_size, int time_step, int input_dim, int hid, 
                      const float* w_x, const float* w_h, const float* b, const float* x, const float* h_0, const float* c_0, 
               /*out*/float *o_t, float *f_t, float *i_t, float* c_wave_t, //hid * batch_size
                      float* c_t, float* h){//time_Step * hid * batch_size
    //printf("x=%f\n", x[0]);
    //printf("x=%f\n", x[1]);
    //printf("h_0=%f\n", h_0[0]);
    //printf("c_0=%f\n", c_0[0]);
    //printf("w_fx = %f\n", w_x[0]);
    //printf("w_ix = %f\n", w_x[1]);
    //printf("w_cx = %f\n", w_x[2]);
    //printf("w_ox = %f\n", w_x[3]);
    //printf("w_fh = %f\n", w_h[0]);
    //printf("w_ih = %f\n", w_h[1]);
    //printf("w_ch = %f\n", w_h[2]);
    //printf("w_oh = %f\n", w_h[3]);
    //printf("bf = %f\n", b[0]);
    //printf("bi = %f\n", b[1]);
    //printf("bc = %f\n", b[2]);
    //printf("bo = %f\n\n", b[3]);
    //global
    const float** A = (const float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    const float** B = (const float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    float** C = (float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    float* x_temp = (float*)mkl_malloc(time_step * 4 * batch_size * hid * sizeof (float), 64);
    
    int i,j,p;
    // w_x * x
    MKL_INT m[1]; 
    MKL_INT n[1]; 
    MKL_INT k[1]; 
    
    MKL_INT lda[1]; 
    MKL_INT ldb[1]; 
    MKL_INT ldc[1]; 
    
    CBLAS_TRANSPOSE transA[1]; 
    CBLAS_TRANSPOSE transB[1]; 
    
    float alpha[1]; 
    float beta[1]; 
    MKL_INT size_per_grp[1]; 

    m[0] = hid;
    k[0] = input_dim;
    n[0] = batch_size;
    
    lda[0] = k[0]; 
    ldb[0] = n[0]; 
    ldc[0] = n[0]; 
    
    transB[0] = CblasNoTrans; 
    transA[0] = CblasNoTrans; 
    
    alpha[0] = 1.0; 
    if (b == NULL) {
        beta[0] = 0.0;
    }
    else {
        beta[0] = 1.0;
        #pragma omp parallel for 
        for (i = 0; i < time_step; i++) { 
            for (j = 0; j < batch_size; j++) { 
                for (p = 0; p < hid; p++) { 
                    size_t offset0 = i * batch_size * hid + j * hid + p; 
                    size_t offset1 = (i + time_step) * batch_size * hid + j * hid + p; 
                    size_t offset2 = (i + 2 * time_step) * batch_size * hid + j * hid + p; 
                    size_t offset3 = (i + 3 * time_step) * batch_size * hid + j * hid + p; 
        
                    x_temp[offset0] = b[p]; 
                    x_temp[offset1] = b[p + hid]; 
                    x_temp[offset2] = b[p + 2 * hid]; 
                    x_temp[offset2] = b[p + 3 * hid]; 
                } 
            } 
        } 
    }
    size_per_grp[0] = 4 * time_step;

    if (NULL == A || NULL == B || NULL == C || NULL == x_temp) {
        printf( "\n ERROR: malloc global buffers failed \n\n");
        return;
    }
    #pragma omp parallel for 
    for (i = 0; i < time_step; i++) { 
        A[i] = w_x;                                       // w_fx
        A[i + time_step] = w_x + input_dim * hid;         // w_ix
        A[i + 2 * time_step] = w_x + 2 * input_dim * hid; // w_cx 
        A[i + 3 * time_step] = w_x + 3 * input_dim * hid; // w_ox 
    
        B[i] = x + i * k[0] * n[0]; 
        B[i + time_step] = B[i]; 
        B[i + 2 * time_step] = B[i]; 
        B[i + 3 * time_step] = B[i]; 
    
        C[i] = x_temp + i * m[0] * n[0]; 
        C[i + time_step] = x_temp + (i + time_step) * m[0] * n[0]; 
        C[i + 2 * time_step] = x_temp + (i + 2 * time_step) * m[0] * n[0]; 
        C[i + 3 * time_step] = x_temp + (i + 3 * time_step) * m[0] * n[0]; 
    } 
    cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp); 

    // loop on step
    m[0] = hid;
    k[0] = hid;
    n[0] = batch_size;
    
    beta[0] = 1.0;

    lda[0] = k[0]; 
    ldb[0] = n[0]; 
    ldc[0] = n[0]; 
    size_per_grp[0] = 4;
    
    A[0] = w_h;                //w_fh
    A[1] = w_h + hid * hid;    //w_ih
    A[2] = w_h + 2 * hid * hid;//w_ch
    A[3] = w_h + 3 * hid * hid;//w_oh
    
    B[0] = h_0;
    B[1] = h_0;
    B[2] = h_0;
    B[3] = h_0;

    size_t mn = m[0] * n[0];
    #pragma omp parallel for
    for (j = 0; j < mn; j++) {
        c_t[j] = c_0[j];
    }

    for (i = 0; i < time_step; i++) {
        // f,i,c_wave,o
        C[0] = x_temp + i * m[0] * n[0];
        C[1] = x_temp + (i + time_step) * m[0] * n[0];
        C[2] = x_temp + (i + 2 * time_step) * m[0] * n[0];
        C[3] = x_temp + (i + 3 * time_step) * m[0] * n[0];

        cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp);

        // sigmoid for f,i,o, tanh for c_wave
        #pragma omp parallel for
        for (j = 0; j < mn; j++) {
            int index = i * mn + j;
            float exp_f = exp((float)(C[0][j]));
            float exp_i = exp((float)(C[1][j]));
            c_wave_t[index] = tanh((float)(C[2][j]));
            float exp_o = exp((float)(C[3][j]));
            f_t[index] = exp_f / ((float)1.0 + exp_f);        
            i_t[index] = exp_i / ((float)1.0 + exp_i);
            o_t[index] = exp_o / ((float)1.0 + exp_o);
        }
        //c
        float c_tm1;
        #pragma omp parallel for 
        for (j = 0; j < mn; j++) { 
            int index = i * mn + j;
            if(i == 0) 
                c_tm1 = c_0[j];
            else
                c_tm1 = c_t[index - mn];
            c_t[index] = (float)((float)(f_t[index]) * (float)(c_tm1) + (float)(i_t[index]) * (float)(c_wave_t[index])); 
        }
        //h:all time_step
        float* y_ptr = NULL;
        y_ptr = h + i * batch_size * hid;
        #pragma omp parallel for
        for (j = 0; j < mn; j++) {
            int index = i * mn + j;
            y_ptr[j] = (float)(o_t[index]) * tanh((float)(c_t[index]));
        }
        // update
        B[0] = y_ptr;
        B[1] = B[0];
        B[2] = B[0];
        B[3] = B[0];
    }
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
    mkl_free(x_temp);
}

void  LSTM_backward(int batch_size, int time_step, int input_dim, int hid, 
                    const float* w_x, const float* w_h, const float* b, const float* x, const float* h_0, const float* c_0,//same with forward input
                    float *ho, float *hf, float *hi, float* hc, float* c, float* h,//forward output: time_step * hid * batch_size
             /*out*/float* dwfx, float* dwix, float* dwcx, float* dwox,
                    float* dwfh, float* dwih, float* dwch, float* dwoh, 
                    float* dbf, float* dbi, float* dbc, float* dbo, 
                    float* dx){
    //global
    const float** A = (const float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    const float** B = (const float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    float** C = (float**)mkl_malloc(4 * time_step * sizeof (float*), 64);
    memset(dwfx, 0, sizeof(float) * hid * input_dim);
    memset(dwix, 0, sizeof(float) * hid * input_dim);
    memset(dwcx, 0, sizeof(float) * hid * input_dim);
    memset(dwox, 0, sizeof(float) * hid * input_dim);
    memset(dwfh, 0, sizeof(float) * hid * hid);
    memset(dwih, 0, sizeof(float) * hid * hid);
    memset(dwch, 0, sizeof(float) * hid * hid);
    memset(dwoh, 0, sizeof(float) * hid * hid);
    memset(dbf, 0, sizeof(float) * hid);
    memset(dbi, 0, sizeof(float) * hid);
    memset(dbc, 0, sizeof(float) * hid);
    memset(dbo, 0, sizeof(float) * hid);
    
    int i,j;
    MKL_INT m[1]; 
    MKL_INT n[1]; 
    MKL_INT k[1]; 
    
    MKL_INT lda[1]; 
    MKL_INT ldb[1]; 
    MKL_INT ldc[1]; 
    
    CBLAS_TRANSPOSE transA[1]; 
    CBLAS_TRANSPOSE transB[1]; 
    
    float alpha[1]; 
    float beta[1]; 
    MKL_INT size_per_grp[1]; 

    //from last timestep
    float *dh = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64); 
    float *dc = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64);
    
    int max_size = max(batch_size, input_dim);
    max_size = max(max_size, hid);
    float *x_temp = (float*)mkl_malloc(4 * time_step * hid * max_size * sizeof (float), 64); 
    
    float *dh_next = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64); 
    float *dc_next = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64);
    memset(dh_next, 0, sizeof(float) * hid * batch_size);
    memset(dc_next, 0, sizeof(float) * hid * batch_size);
    //temp mem
    float *dhf = (float*)mkl_malloc(time_step * hid * batch_size * sizeof (float), 64); 
    float *dhi = (float*)mkl_malloc(time_step * hid * batch_size * sizeof (float), 64); 
    float *dhc = (float*)mkl_malloc(time_step * hid * batch_size * sizeof (float), 64); 
    float *dho = (float*)mkl_malloc(time_step * hid * batch_size * sizeof (float), 64); 
    
    float *dhhf = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64); 
    float *dhhi = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64); 
    float *dhhc = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64); 
    float *dhho = (float*)mkl_malloc(hid * batch_size * sizeof (float), 64); 
    
    float *dxf = (float*)mkl_malloc(time_step * input_dim * batch_size * sizeof (float), 64);
    float *dxi = (float*)mkl_malloc(time_step * input_dim * batch_size * sizeof (float), 64);
    float *dxc = (float*)mkl_malloc(time_step * input_dim * batch_size * sizeof (float), 64);
    float *dxo = (float*)mkl_malloc(time_step * input_dim * batch_size * sizeof (float), 64);
    //cache: hf hi hc ho c, c=[c_0, all c_t]i
    //calculate all gf, gi, gc_wave, go
    // loop on step
    m[0] = hid;
    k[0] = hid;
    n[0] = batch_size;
    
    beta[0] = 0.0;

    lda[0] = m[0]; 
    ldb[0] = n[0]; 
    ldc[0] = n[0]; 
    transA[0] = CblasTrans; 
    transB[0] = CblasNoTrans; 
    
    alpha[0] = 1.0; 
    size_per_grp[0] = 4;
    
    A[0] = w_h;                //w_fh
    A[1] = w_h + hid * hid;    //w_ih
    A[2] = w_h + 2 * hid * hid;//w_ch
    A[3] = w_h + 3 * hid * hid;//w_oh
    
    C[0] = dhhf;
    C[1] = dhhi;
    C[2] = dhhc;
    C[3] = dhho;
    size_t bh = batch_size * hid;
    size_t tbi = batch_size * input_dim * time_step;
    size_t ib = input_dim * batch_size;
    size_t hh = hid * hid;
    for(i = time_step - 1; i >= 0; i--) {
        int kk = i * bh;
        for(j = 0; j < bh; j++ ) {
            int index = kk + j;
            float c_old;
            if(i != 0) 
                c_old = c[index - bh];
            else
                c_old = c_0[j];
            float tanh_c = tanh(c[index]);
            if(i == time_step - 1) 
                dh[j] = 1.0 + dh_next[j];
            else
                dh[j] = dh_next[j];
            //printf("error! %d, ho[index]=%f, tanh_c=%f, dh[j]=%f\n", index, ho[index], tanh_c, dh[j]);
            dho[index] = ho[index] * (1.0 - ho[index]) * tanh_c * dh[j];
            dc[j] = ho[index] * dh[j] * (1.0 - tanh_c * tanh_c) + dc_next[j];
            dhf[index] = hf[index] * (1.0 - hf[index]) * c_old * dc[j];
            dhi[index] = hi[index] * (1.0 - hi[index]) * hc[index] * dc[j];
            dhc[index] = (1.0 - hc[index] * hc[index]) * hi[index] * dc[j];

            
            //printf("time_step=%d\n", i);
            //printf("hf[%d]=%f\n", index, hf[index]);
            //printf("hi[%d]=%f\n", index, hi[index]);
            //printf("hc[%d]=%f\n", index, hc[index]);
            //printf("ho[%d]=%f\n", index, ho[index]);
            //
            //printf("dh_next[%d]=%f\n", j, dh_next[j]);
            //printf("dh[%d]=%f\n", j, dh[j]);
            //printf("dc[%d]=%f\n", j, dc[j]);
            //printf("dhf[%d]=%f\n", index, dhf[index]);
            //printf("dhi[%d]=%f\n", index, dhi[index]);
            //printf("dhc[%d]=%f\n", index, dhc[index]);
            //printf("dho[%d]=%f\n\n", index, dho[index]);
        }
        B[0] = dhf + kk;
        B[1] = dhi + kk;
        B[2] = dhc + kk;
        B[3] = dho + kk;
        cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp);
    
        //calculate dbf, dbi, dbc, dbo
        for(j = 0; j < bh; j++ ) {
            int index = kk + j;
            dh_next[j] = dhhf[j] + dhhi[j] + dhhc[j] + dhho[j];
            //printf("dhhf=%f\n", dhhf[j]);
            //printf("dhhi=%f\n", dhhi[j]);
            //printf("dhhc=%f\n", dhhc[j]);
            //printf("dhho=%f\n", dhho[j]);
            dc_next[j] = hf[index] * dc[j];
            //printf("dh_next[%d]=%f\n", j,dh_next[j]);
            //printf("dc_next[%d]=%f\n\n\n\n", j,dc_next[j]);
        }
        for(j = 0; j < hid; j++) {
            for(int p = 0; p < batch_size; p++) {
                int index = kk + j * batch_size + p;
                dbf[j] += dhf[index];
                dbi[j] += dhi[index];
                dbc[j] += dhc[index];
                dbo[j] += dho[index];
            }
        } 
    }
    //calculate dwfx, dwix, dwcx, dwox
    m[0] = hid;
    k[0] = batch_size;
    n[0] = input_dim;

    lda[0] = k[0];
    ldb[0] = k[0];
    ldc[0] = n[0];
    transA[0] = CblasNoTrans;
    transB[0] = CblasTrans;
    
    size_per_grp[0] = 4 * time_step;
    for (i = 0; i < time_step; i++) {
        A[i] = dhf + i * bh;                 
        A[i + time_step] = dhi + i * bh;    
        A[i + 2 * time_step] = dhc + i * bh; 
        A[i + 3 * time_step] = dho + i * bh; 
    
        B[i] = x + i * input_dim * batch_size; 
        B[i + time_step] = B[i]; 
        B[i + 2 * time_step] = B[i]; 
        B[i + 3 * time_step] = B[i]; 
    
        C[i] = x_temp + i * hid * input_dim;
        C[i + time_step] = x_temp + (i + time_step) * hid * input_dim; 
        C[i + 2 * time_step] = x_temp + (i + 2 * time_step) * hid * input_dim; 
        C[i + 3 * time_step] = x_temp + (i + 3 * time_step) * hid * input_dim; 
    }
    cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp);
    for(i = 0; i < hid * input_dim; i++) {
        for(j = 0; j < time_step; j++) {
            int index = j * hid * input_dim + i;
            //dwfx[i] += dwfx_allt[index];
            //dwix[i] += dwix_allt[index];
            //dwcx[i] += dwcx_allt[index];
            //dwox[i] += dwox_allt[index];
            dwfx[i] += C[j][i];
            dwix[i] += C[j + time_step][i];
            dwcx[i] += C[j + 2 * time_step][i];
            dwox[i] += C[j + 3 * time_step][i];
            //printf("%f, %f\n", dwfx_allt[index], C[j][i]);
            //printf("%f, %f\n", dwix_allt[index], C[j + time_step][i]);
            //printf("%f, %f\n", dwcx_allt[index], C[j + 2 * time_step][i]);
            //printf("%f, %f\n", dwox_allt[index], C[j + 3 * time_step][i]);
            //printf("%f\n", C[j][i]);
            //printf("%f\n", C[j + time_step][i]);
            //printf("%f\n", C[j + 2 * time_step][i]);
            //printf("%f\n", C[j + 3 * time_step][i]);
        }
    }
    //calculate dwfh, dwih, dwch, dwoh
    m[0] = hid;
    k[0] = batch_size;
    n[0] = hid;

    lda[0] = k[0];
    ldb[0] = k[0];
    ldc[0] = n[0];
    transA[0] = CblasNoTrans;
    transB[0] = CblasTrans;
    
    size_per_grp[0] = 4 * time_step;
    for (i = 0; i < time_step; i++) {
        A[i] = dhf + i * bh;                 
        A[i + time_step] = dhi + i * bh;    
        A[i + 2 * time_step] = dhc + i * bh; 
        A[i + 3 * time_step] = dho + i * bh; 
   
        if(i == 0) {
            B[i] = h_0; 
            B[i + time_step] = B[i]; 
            B[i + 2 * time_step] = B[i]; 
            B[i + 3 * time_step] = B[i]; 
        }    
        else {
            B[i] = h + (i - 1) * bh; 
            B[i + time_step] = B[i]; 
            B[i + 2 * time_step] = B[i]; 
            B[i + 3 * time_step] = B[i]; 
        } 
        C[i] = x_temp + i * hh;
        C[i + time_step] = x_temp + (i + time_step) * hh; 
        C[i + 2 * time_step] = x_temp + (i + 2 * time_step) * hh; 
        C[i + 3 * time_step] = x_temp + (i + 3 * time_step) * hh; 
    } 
    cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp);
    for(i = 0; i < hid * hid; i++) {
        for(j = 0; j < time_step; j++) {
            int index = j * hid * hid + i;
            dwfh[i] += C[j][i];
            dwih[i] += C[j + time_step][i];
            dwch[i] += C[j + 2 * time_step][i];
            dwoh[i] += C[j + 3 * time_step][i];
            //printf("time_step=%d\n", j);
            //printf("dwfh_allt=%f\n", dwfh_allt[index]);
            //printf("dwih_allt=%f\n", dwih_allt[index]);
            //printf("dwch_allt=%f\n", dwch_allt[index]);
            //printf("dwoh_allt=%f\n\n", dwoh_allt[index]);
        }
    }

    //calculate dx
    m[0] = input_dim;
    k[0] = hid;
    n[0] = batch_size;
    
    lda[0] = m[0]; 
    ldb[0] = n[0]; 
    ldc[0] = n[0]; 
    transA[0] = CblasTrans;
    transB[0] = CblasNoTrans;
    size_per_grp[0] = 4 * time_step;
    for (i = 0; i < time_step; i++) { 
        A[i] = w_x;                                       // w_fx
        A[i + time_step] = w_x + input_dim * hid;         // w_ix
        A[i + 2 * time_step] = w_x + 2 * input_dim * hid; // w_cx 
        A[i + 3 * time_step] = w_x + 3 * input_dim * hid; // w_ox 
    
        B[i] = dhf + i * bh; 
        B[i + time_step] = dhi + i * bh; 
        B[i + 2 * time_step] = dhc + i * bh; 
        B[i + 3 * time_step] = dho + i * bh; 
    
        C[i] = dxf + i * ib;
        C[i + time_step] = dxi + i * ib; 
        C[i + 2 * time_step] = dxc + i * ib; 
        C[i + 3 * time_step] = dxo + i * ib; 
    }
    cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 1, size_per_grp); 
    for(i = 0; i < tbi; i++) {
        dx[i] = dxf[i] + dxi[i] + dxc[i] + dxo[i];
    }
    mkl_free(dh); 
    mkl_free(dc); 
    mkl_free(dh_next); 
    mkl_free(dc_next); 
    mkl_free(dhf); 
    mkl_free(dhi); 
    mkl_free(dhc); 
    mkl_free(dho); 
    mkl_free(dhhf); 
    mkl_free(dhhi); 
    mkl_free(dhhc); 
    mkl_free(dhho); 
    mkl_free(dxf);
    mkl_free(dxi);
    mkl_free(dxc);
    mkl_free(dxo);
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
}
#ifdef __cplusplus
}
#endif
