Steps to do this:

1. Clear Stale files -- this does not work too well -- it only looks at some weird directory
2. Make sure the base model is loaded in onnx 
3. Get individual sub-blocks -- save them
4. Generate sub-block jacobians
    * if all jacobians -- stitch together
    * if individual jacobians -- save one by one


Dynamic Batching is



Go to makespace -- get the headphones off
Eat

Then unify all code for subgraphs and pair jaocibans
  Make sub function that creates a model -- done
  Make sub function that extracts subgraphs -- done
  Make sub function that generates the gradient of a subgraphs -- done
      Make sub function that creates the loss -- done
      Make sub function that creates the jacobian mask -- done
  

  There are two cases now where I would like to improve
    - first getting a jacobian of a graph without extracting
      - All jacobians with real activations
    - getting a twin loss for adversarial things
      - getting a loss function which takes a specific input


BLAS checking
