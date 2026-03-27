# Ar2Can: An Architect and an Artist Leveraging a Canvas for Multi-Human Generation

## Abstract

Despite recent advances in personalized image generation, existing models consistently fail to produce reliable multi-human scenes, often merging facial identities or generating incorrect person counts. These failures stem from entangling spatial planning and identity rendering within a single-stage generation process.

We present **Ar2Can**, a novel two-stage framework that explicitly disentangles spatial planning from identity-preserving rendering for multi-human image generation. Ar2Can decomposes the task into an **Architect**, which predicts structured spatial layouts specifying where each person should appear, and an **Artist**, which synthesizes photorealistic images guided by these layouts. The Artist is optimized using **Group Relative Policy Optimization (GRPO)** with a compositional reward that includes spatially-grounded face matching via Hungarian assignment, identity similarity, count accuracy, pose alignment, and perceptual image quality.

Ar2Can supports multiple Architect designs with complementary trade-offs between accuracy and efficiency, and introduces token sharing and dropping strategies to improve inference efficiency and handle occlusions. Evaluated on the MultiHuman-Testbench, Ar2Can achieves state-of-the-art performance in both identity preservation and count accuracy while maintaining high perceptual quality. Notably, the method relies primarily on synthetic data and does not require real multi-human training images, demonstrating a scalable and practical solution for personalized multi-human image generation.
