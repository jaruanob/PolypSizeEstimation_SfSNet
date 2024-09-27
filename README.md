# Estimating polyp size from a single colonoscopy image using a shape-from-shading model

Josué Ruano(a), Diego Bravo(a), Diana Giraldo(a,b), Martín Gómez(c), Fabio A. González(d), Antoine Manzanera(e) and Eduardo Romero(a)

(a) Computer Imaging and Medical Applications Laboratory (CIM@LAB)
(b) imec-Vision Lab, Department of Physics, University of Antwerp, Antwerp, Belgium
(c) Hospital Universitario Nacional de Colombia, Unidad de Gastroenterologı́a, Bogotá, Colombia
(d) Machine Learning, Perception and Discovery Lab (MindLab)
(e) Unité d’Informatique et d’Ingénierie des Systèmes, ENSTA-Institut Polytechnique de Paris, France
(a,c,d) Universidad Nacional de Colombia, Bogotá, Colombia

<img src="pipeline_isbi.png?raw=True" width="800px" style="margin:0px 0px"/>

# Previous work
This method uses a depth estimation model exaustively validated, Shape-from-Shading Network (SfSNet), and published as:

Ruano, J., Gómez, M., Romero, E., & Manzanera, A. (2024). Leveraging a realistic synthetic database to learn Shape-from-Shading for estimating the colon depth in colonoscopy images. Computerized Medical Imaging and Graphics, [DOI](https://doi.org/10.1016/j.compmedimag.2024.102390)

Repository: [SfSNet](https://github.com/Cimalab-unal/ColonDepthEstimation) 

# Results

Name of images, ground truth size, and estimations are in a CSV for each collection:

  1. Synthetic: Size_estimation_Synthetic.csv
  2. Real: size_estimationSUNDb_1img.csv

# Synthetic colonoscopy database

At the link below you can request access to the database.

https://forms.gle/USQkvguACcNTeGH26

When you have access to the database, you will find a zip file with name "SyntheticDatabase_testingset_PolypSize.zip". This file contains the synthetic videos used for testing this method. Each folder provides depth maps (z), RGB images (img), and binary mask with segmented polyps (mask).
