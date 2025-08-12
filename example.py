import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils



# Model path: /nfshomes/yinghy/.cache/torch/hub/facebookresearch_dinov2_main



# Load a pipeline from a model folder or a Hugging Face model hub.
# pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline = TrellisImageTo3DPipeline.from_pretrained(f"{os.getcwd()}/ckpts/models--microsoft--TRELLIS-image-large/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96")   # Using local path, make sure there is a pipeline.json under the path
pipeline.cuda()

# Load an image
# image_path = "assets/example_image/T.png"
# image_path = "assets/example_image/4_colors.png"
# image_path = "assets/example_image/30_colors.png"
# image_path = "assets/example_image/5225_10_colors.png"
# image_path = "assets/example_image/5084_26_colors.png"
# image_path = "assets/example_image/4926_17_colors.png"
# image_path = "assets/example_image/3884_1_colors.png"
# image_path = "assets/example_image/3619_47_colors.png"
# image_path = "assets/example_image/3598_34_colors.png"
# image_path = "assets/example_image/3525_13_colors.png"
# image_path = "assets/example_image/2954_22_colors.png"

img_list = [
    # "assets/example_image/1911_3_colors.png",
    # "assets/example_image/1614_24_colors.png",
    # "assets/example_image/1314_37_colors.png",
    # "assets/example_image/0952_2_colors.png",
    # "assets/example_image/0568_2_colors.png"
    # "assets/example_image/kangoroo2.png"
    # "assets/example_image/thinker.png"
    # "assets/example_image/cad1.png"
    # "assets/example_image/cad2.png",
    "assets/example_image/cad3.png"
    ]

for image_path in img_list:

    image = Image.open(image_path)
    image_name = os.path.basename(image_path).split('.')[0]

    save_path = f"output/{image_name}"
    os.makedirs(save_path)

    # Run the pipeline
    outputs = pipeline.run(
        image,
        seed=1,
        # Optional parameters
        # sparse_structure_sampler_params={
        #     "steps": 12,
        #     "cfg_strength": 7.5,
        # },
        # slat_sampler_params={
        #     "steps": 12,
        #     "cfg_strength": 3,
        # },
    )
    # outputs is a dictionary containing generated 3D assets in different formats:
    # - outputs['gaussian']: a list of 3D Gaussians
    # - outputs['radiance_field']: a list of radiance fields
    # - outputs['mesh']: a list of meshes

    # Render the outputs
    video = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave(f"{save_path}/sample_gs.mp4", video, fps=30)
    video = render_utils.render_video(outputs['radiance_field'][0])['color']
    imageio.mimsave(f"{save_path}/sample_rf.mp4", video, fps=30)
    video = render_utils.render_video(outputs['mesh'][0])['normal']
    imageio.mimsave(f"{save_path}/sample_mesh.mp4", video, fps=30)

    # GLB files can be extracted from the outputs
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        # Optional parameters
        simplify=0.95,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
    )
    glb.export(f"{save_path}/sample.glb")

    # Save Gaussians as PLY files
    outputs['gaussian'][0].save_ply(f"{save_path}/sample.ply")
