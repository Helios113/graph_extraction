Steps to do this:

1. Clear Stale files -- this does not work too well -- it only looks at some weird directory
2. Make sure the base model is loaded in onnx 
3. Get individual sub-blocks -- save them
4. Generate sub-block jacobians
    * if all jacobians -- stitch together
    * if individual jacobians -- save one by one


Dynamic Batching is




<!-- 1. Tell the Gradient block to generate an accumulator -->
<!-- 2. Add optimizer to the gradient -- we can do this manually -->
3. BLAS checking



