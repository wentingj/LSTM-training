#include "/home/wentingj/tensorflow/tensorflow/core/framework/op.h"
#include "/home/wentingj/tensorflow/tensorflow/core/framework/shape_inference.h"

#include "/home/wentingj/tensorflow/tensorflow/core/framework/op_kernel.h"

#include "lstm_training_src.cc"

#include "mkl.h"
#include "math.h"
#include<sys/time.h>
using namespace std;
using namespace tensorflow;

REGISTER_OP("IntelLstmTrain")
    .Input("x: float")
    .Input("w_x: float")
    .Input("w_h: float")
    .Input("b: float")
    .Input("h_0: float")
    .Input("c_0: float")
    .Output("dall: float")
    /*.Output("dwfx: float")
    .Output("dwix: float")
    .Output("dwcx: float")
    .Output("dwox: float")
    .Output("dwfh: float")
    .Output("dwih: float")
    .Output("dwch: float")
    .Output("dwoh: float")
    .Output("dbf: float")
    .Output("dbi: float")
    .Output("dbc: float")
    .Output("dbo: float")
    .Output("dx: float")*/
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
     //c->set_output(0,c->input(0));
     return Status::OK();
    });


class IntelLstmTrainOp : public OpKernel {
 public:
   explicit IntelLstmTrainOp(OpKernelConstruction* context) : OpKernel(context) {}
     void Compute(OpKernelContext* context) override {
         int i,j,p;
         //x
         const Tensor* input_tensor = &context->input(0);
         auto input_flat = input_tensor->flat<float>();
         auto x = input_flat.data();
         //printf( " run to %s on line %d\n " ,__FILE__, __LINE__);
         
         //w_x
         const Tensor* w_x_tensor = &context->input(1);
         auto w_x_flat = w_x_tensor->flat<float>();
         auto w_x = w_x_flat.data();

         //w_h
         const Tensor* w_h_tensor = &context->input(2);
         auto w_h_flat = w_h_tensor->flat<float>();
         auto w_h = w_h_flat.data();

         //b
         const Tensor* b_tensor = &context->input(3);
         auto b_flat = b_tensor->flat<float>();
         auto b = b_flat.data();

         //h_0
         const Tensor* h_0_tensor = &context->input(4);
         auto h_0_flat = h_0_tensor->flat<float>();
         auto h_0 = h_0_flat.data();
       
         //c_0
         const Tensor* c_0_tensor = &context->input(5);
         auto c_0_flat = c_0_tensor->flat<float>();
         auto c_0 = c_0_flat.data();

         int max_len = 128;//max timestep
         int batch_size = input_tensor->shape().dim_size(2);
         int time_step = input_tensor->shape().dim_size(0);
         int input_dim = input_tensor->shape().dim_size(1);
         int hid = w_h_tensor->shape().dim_size(1);

         //output
         TensorShape shape_dall = TensorShape({hid * input_dim * 4 + hid * hid * 4 + hid * 4 + time_step * batch_size * input_dim});
         Tensor* dall_tensor = NULL;
         OP_REQUIRES_OK(context, context->allocate_output(0, shape_dall, &dall_tensor));
         auto dall_flat = dall_tensor->flat<float>();
         auto dall = dall_flat.data();

         /*TensorShape shape_dwfx = TensorShape({hid, input_dim});
         Tensor* dwfx_tensor = NULL;
         OP_REQUIRES_OK(context, context->allocate_output(0, shape_dwfx, &dwfx_tensor));
         auto dwfx_flat = dwfx_tensor->flat<float>();
         auto dwfx = dwfx_flat.data();

         TensorShape shape_dwix = TensorShape({hid, input_dim});
         Tensor* dwix_tensor = NULL;
         OP_REQUIRES_OK(context, context->allocate_output(1, shape_dwix, &dwix_tensor));
         auto dwix_flat = dwix_tensor->flat<float>();
         auto dwix = dwix_flat.data();

         TensorShape shape_dwcx = TensorShape({hid, input_dim});
         Tensor* dwcx_tensor = NULL;
         OP_REQUIRES_OK(context, context->allocate_output(2, shape_dwcx, &dwcx_tensor));
         auto dwcx_flat = dwcx_tensor->flat<float>();
         auto dwcx = dwcx_flat.data();
         
         TensorShape shape_dwox = TensorShape({hid, input_dim});
         Tensor* dwox_tensor = NULL;
         OP_REQUIRES_OK(context, context->allocate_output(3, shape_dwox, &dwox_tensor));
         auto dwox_flat = dwox_tensor->flat<float>();
         auto dwox = dwox_flat.data();
         //h
         TensorShape shape_dwfh = TensorShape({hid, hid});
         Tensor* dwfh_tensor = NULL;
         OP_REQUIRES_OK(context, context->allocate_output(4, shape_dwfh, &dwfh_tensor));
         auto dwfh_flat = dwfh_tensor->flat<float>();
         auto dwfh = dwfh_flat.data();

         TensorShape shape_dwih = TensorShape({hid, hid});
         Tensor* dwih_tensor = NULL;
         OP_REQUIRES_OK(context, context->allocate_output(5, shape_dwih, &dwih_tensor));
         auto dwih_flat = dwih_tensor->flat<float>();
         auto dwih = dwih_flat.data();

         TensorShape shape_dwch = TensorShape({hid, hid});
         Tensor* dwch_tensor = NULL;
         OP_REQUIRES_OK(context, context->allocate_output(6, shape_dwch, &dwch_tensor));
         auto dwch_flat = dwch_tensor->flat<float>();
         auto dwch = dwch_flat.data();
         
         TensorShape shape_dwoh = TensorShape({hid, hid});
         Tensor* dwoh_tensor = NULL;
         OP_REQUIRES_OK(context, context->allocate_output(7, shape_dwoh, &dwoh_tensor));
         auto dwoh_flat = dwoh_tensor->flat<float>();
         auto dwoh = dwoh_flat.data();
         //b
         TensorShape shape_dbf = TensorShape({hid, batch_size});
         Tensor* dbf_tensor = NULL;
         OP_REQUIRES_OK(context, context->allocate_output(8, shape_dbf, &dbf_tensor));
         auto dbf_flat = dbf_tensor->flat<float>();
         auto dbf = dbf_flat.data();

         TensorShape shape_dbi = TensorShape({hid, batch_size});
         Tensor* dbi_tensor = NULL;
         OP_REQUIRES_OK(context, context->allocate_output(9, shape_dbi, &dbi_tensor));
         auto dbi_flat = dbi_tensor->flat<float>();
         auto dbi = dbi_flat.data();

         TensorShape shape_dbc = TensorShape({hid, batch_size});
         Tensor* dbc_tensor = NULL;
         OP_REQUIRES_OK(context, context->allocate_output(10, shape_dbc, &dbc_tensor));
         auto dbc_flat = dbc_tensor->flat<float>();
         auto dbc = dbc_flat.data();
         
         TensorShape shape_dbo = TensorShape({hid, batch_size});
         Tensor* dbo_tensor = NULL;
         OP_REQUIRES_OK(context, context->allocate_output(11, shape_dbo, &dbo_tensor));
         auto dbo_flat = dbo_tensor->flat<float>();
         auto dbo = dbo_flat.data();

         TensorShape shape_dx = TensorShape({time_step, input_dim, batch_size});
         Tensor* dx_tensor = NULL;
         OP_REQUIRES_OK(context, context->allocate_output(12, shape_dx, &dx_tensor));
         auto dx_flat = dx_tensor->flat<float>();
         auto dx = dx_flat.data();*/

         //share from forward
         float* hf = (float*)mkl_malloc(time_step * batch_size * hid * sizeof (float), 64);
         float* hi = (float*)mkl_malloc(time_step * batch_size * hid * sizeof (float), 64);
         float* hc = (float*)mkl_malloc(time_step * batch_size * hid * sizeof (float), 64);
         float* ho = (float*)mkl_malloc(time_step * batch_size * hid * sizeof (float), 64);
         float* c = (float*)mkl_malloc(time_step * batch_size * hid * sizeof (float), 64);

         //forward output
         float* h = (float*)mkl_malloc(time_step * hid * batch_size * sizeof (float), 64);
         memset(h, 0, sizeof(float) * time_step * hid * batch_size);

         LSTM_batch_gemm(batch_size, time_step, input_dim, hid,
                         w_x, w_h, b, x, h_0, c_0,
                         ho, hf, hi, hc, //hid * batch_size
                         c, h);//time_Step * hid * batch_size
         LSTM_backward(batch_size, time_step, input_dim, hid, 
                    w_x, w_h, b, x, h_0, c_0,//same with forward input
                    ho, hf, hi, hc, c, h,//forward output: time_step * hid * batch_size
                    dall + hid * input_dim * 2,
                    dall,
                    dall + hid * input_dim,
                    dall + hid * input_dim * 3,//dwxf,i,c,o

                    dall + hid * input_dim * 4 + hid * hid * 2, 
                    dall + hid * input_dim * 4, 
                    dall + hid * input_dim * 4 + hid * hid, 
                    dall + hid * input_dim * 4 + hid * hid * 3,//dwhf,i,c,o

                    dall + hid * input_dim * 4 + hid * hid * 4 + hid * 2,
                    dall + hid * input_dim * 4 + hid * hid * 4, 
                    dall + hid * input_dim * 4 + hid * hid * 4 + hid, 
                    dall + hid * input_dim * 4 + hid * hid * 4 + hid * 3,//dbf,i,c,o 
                    dall + hid * input_dim * 4 + hid * hid * 4 + hid * 4);//dx
                    //icfo    
         mkl_free(hf);
         mkl_free(hi);
         mkl_free(hc);
         mkl_free(ho);
         mkl_free(c);
         mkl_free(h);
    }
};

REGISTER_KERNEL_BUILDER(Name("IntelLstmTrain").Device(DEVICE_CPU), IntelLstmTrainOp);
