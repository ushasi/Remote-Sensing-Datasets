# Remote-Sensing-Datasets
A list of radar and optical satellite datasets for detection, classification, semantic segmentation and instance segmentation tasks.


![GitHub latest commit](https://img.shields.io/maintenance/yes/2023?style=plastic&service=github)
![GitHub latest commit](https://img.shields.io/github/last-commit/ushasi/Remote-Sensing-Datasets?style=plastic&service=github)

# Satellite imagery datasets according to their corresponding tasks.<a name="TOP"></a> 
A list of radar and optical satellite datasets for computer vision and deep learning. Categories: Instance segmentation, object detection, semantic segmentation, scene classification, retrieval, other.

<div align="justify">

*The specific datasets could not be accessed. Parts of information on the repository credits are from [here](https://github.com/jasonmanesis/Satellite-Imagery-Datasets-Containing-Ships/blob/main/README.md#SSDD), [here](https://github.com/chrieke/awesome-satellite-imagery-datasets), and [here]().

# Satellite imagery datasets containing aerial view images.<a name="TOP"></a> 
A list of radar and optical satellite datasets for aerial scene isntance detection, classification, semantic segmentation and instance segmentation tasks.
 
 
 # 1. Scene classification

 
  ## :satellite: Radar Satellite Datasets : 
- [**FUSAR-Ship Dataset v1.0 - 2020, Hou et al.**](#FUSAR-Ship) ↦ Classification 
- [**Statoil/C-CORE Iceberg Classifier Challenge**](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge)
2 categories ship and iceberg, 2-band HH/HV polarization SAR imagery.
 
 
  ## :satellite: Optical Satellite Datasets : 
 - [**Airbus Wind Turbine Patches**](https://www.kaggle.com/airbusgeo/airbus-wind-turbines-patches)    
155k 128x128px image chips with wind turbines (SPOT, 1.5m res.). 
-  [**BigEarthNet: Large-Scale Sentinel-2 Benchmark**](http://bigearth.net) 
Multilabel (CLC) 2018, 590,326 chips from Sentinel-2 L2A scenes.
 - [**WiDS Datathon 2019 : Detection of Oil Palm Plantations**](https://www.kaggle.com/c/widsdatathon2019) 
 Planet satellite imagery (3m res.)., ca. 20k 256 x 256 px. 
 - [**Cactus Aerial Photos**](https://www.kaggle.com/irvingvasquez/cactus-aerial-photos)
17k aerial photos, 13k cactus, 4k non-actus.
 - [**Functional Map of the World Challenge**](https://www.iarpa.gov/challenges/fmow.html) 
63 categories from solar farms to shopping malls, 1 million chips, 4/8 band satellite imagery (0.3m res.).
- [**EuroSAT**](http://madm.dfki.de/downloads)
10 land cover categories, 27k 64x64 pixel chips, 3/16 band Sentinel-2 satellite imagery (10m res.).
- [**AID: Aerial Scene Classification**](https://captain-whu.github.io/AID/) 
10000 aerial images within 30 categories collected from Google Earth imagery.
- [**RESISC45**](https://www.tensorflow.org/datasets/catalog/resisc45) 
45 scene categories, 31,500 images (700 per category, 256x256 px), image chips taken from Google Earth.
 - [**Deepsat: SAT-4/SAT-6 airborne datasets**](https://csc.lsu.edu/~saikat/deepsat/) 
6 land cover categories, 400k 28x28 pixel chips, 4-band RGBNIR aerial imagery (1m res.). 
- [**UC Merced Land Use Dataset**](http://weegee.vision.ucmerced.edu/datasets/landuse.html)
21 land cover categories from agricultural to parkinglot, 100 chips per class, aerial imagery (0.30m res.).
 
 
  ## :satellite: Multi-sensor Satellite Datasets : 
 - [**So2Sat LCZ42**](https://mediatum.ub.tum.de/1454690) *(TUM Munich & DLR, Aug 2018)*   
Local climate zone classification, 400k 32x32 pixel chips covering 42 cities (LCZ42 dataset), Sentinel 1 & Sentinel 2 (both 10m res.), 51 GB   
 - [**Planet: Understanding the Amazon from Space**](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space)  
13 land cover categories + 4 cloud condition categories, 4-band (RGB-NIR) satelitte imagery (5m res.).
 
 
# 2. Retrieval
 
  ## :satellite: Radar Satellite Datasets : coming up!!
 
  ## :satellite: Optical Satellite Datasets : coming up!!
 
  ## :satellite: Multi-sensor Satellite Datasets : coming up!!
 
  ## :satellite: Cross-sensor Datasets : coming up!!
 
 
 
# 3. Instance Segmentation

## 🛰: Radar Satellite Datasets:
                                  
* [**HRSID (High-Resolution SAR Images Dataset) - 2020, Wei et al.**](#HRSID)
* [**SSDD (SAR Ship Detection Dataset) - 2021, Zhang et al.**](#SSDD1)
 
 
## :satellite: Optical Satellite Datasets : 

- [**PASTIS: Panoptic Agricultural Satellite Time Series**](https://github.com/VSainteuf/pastis-benchmark) Sentinel-2 image chip timeseries, panoptic labels (instance index + semantic label for each pixel).
- [**SpaceNet 7: Multi-Temporal Urban Development Challenge**](https://spacenet.ai/sn7-challenge/) 
Monthly building footprints and Planet imagery (4m. res) timeseries.
- [**SpaceNet 4: Off-Nadir Buildings**](https://spacenet.ai/off-nadir-building-detection/)
126k building footprints (Atlanta), 27 WorldView 2 images (0.3m res.).
- [**SpaceNet 2: Building Detection v2**](https://spacenet.ai/spacenet-buildings-dataset-v2/)  
685k building footprints, 3/8band Worldview-3 imagery (0.3m res.). 
- [**SpaceNet 1: Building Detection v1**](https://spacenet.ai/spacenet-buildings-dataset-v1/) 
Building footprints (Rio de Janeiro), 3/8band Worldview-3 imagery (0.5m res.).
- [**RarePlanes: Synthetic Data Takes Flight**](https://aireverie.com/rareplanes) *(CosmiQ Works, A.I.Reverie, June 2020)*   
Synthetic (630k planes, 50k images) and real (14.7k planes, 253 Worldview-3 images (0.3m res.), 122 locations, 22 countries) plane annotations & properties and satellite images. [Tools](https://github.com/aireveries/RarePlanes). Paper: [Shermeyer et al. 2020](https://arxiv.org/abs/2006.02963)
- [**Agriculture-Vision Database & CVPR 2020 challenge**](https://www.agriculture-vision.com/agriculture-vision-2020/dataset) 
21k aerial farmland images (RGB-NIR, 512x512px chips), label masks for 6 field anomaly patterns..
- [**iSAID: Large-scale Dataset for Object Detection in Aerial Images**](https://captain-whu.github.io/iSAID/dataset.html) 
15 categories, 188k instances, object instances and segmentation masks, Google Earth & JL-1 image chips, replaces [DOTA dataset](https://captain-whu.github.io/DOTA/).
- [**xView 2 Building Damage Asessment Challenge**](https://xview2.org) *(DIUx, Nov 2019)* .   
550k building footprints & 4 damage scale categories, 20 global locations and 7 disaster types , Worldview-3 imagery (0.3m res.). 
- **Microsoft BuildingFootprints** [**Canada**](https://github.com/Microsoft/CanadianBuildingFootprints) **&** [**USA**](
https://github.com/microsoft/USBuildingFootprints) **&** [**Uganda/Tanzania**](
https://github.com/microsoft/Uganda-Tanzania-Building-Footprints) **&** [Australia](https://github.com/microsoft/AustraliaBuildingFootprints) *(Microsoft, Mar 2019)*   
 GeoJSON format, delineation based on Bing imagery. 
- [**Airbus Ship Detection Challenge**](https://www.kaggle.com/c/airbus-ship-detection) 
131k ships, 104k train / 88k test image chips, satellite imagery (1.5m res.). 
- [**Open AI Challenge: Tanzania**](https://competitions.codalab.org/competitions/20100#learn_the_details-overview)   
Building footprints & 3 building conditions, RGB UAV imagery - [Link to data](https://docs.google.com/spreadsheets/d/1kHZo2KA0-VtCCcC5tL4N0SpyoxnvH7mLbybZIHZGTfE/edit#gid=0).
- [**CrowdAI Mapping Challenge**](https://www.crowdai.org/challenges/mapping-challenge) 
Buildings footprints, RGB satellite imagery.


## :satellite: Multi-sensor Satellite Datasets : 

- [**SpaceNet: Multi-Sensor All-Weather Mapping**](https://spacenet.ai/sn6-challenge/)  
48k building footprints (enhanced 3DBAG dataset, building height attributes), SAR data (4 polarizations) & Worldview-3.
- **LPIS agricultural field boundaries** [Denmark](https://kortdata.fvm.dk/download/Markblokke_Marker?page=MarkerHistoriske) - [Netherlands](https://www.pdok.nl/introductie/-/article/basisregistratie-gewaspercelen-brp-) - [France](https://www.data.gouv.fr/en/datasets/registre-parcellaire-graphique-rpg-contours-des-parcelles-et-ilots-culturaux-et-leur-groupe-de-cultures-majoritaire/)  
Denmark: 293 crop/vegetation catgeories, 600k parcels. Netherlands: 294 crop/vegetation catgeories, 780k parcels

 


# 4. Object Detection

## :satellite: Radar Satellite Datasets : 

* [**SSDD (SAR Ship Detection Dataset) - 2017, Li et al.**](#SSDD) 
* [**OpenSARship-1.0, 2.0- 2017, Huang et al.**](#OpenSARship-1.0)
* [**SAR-Ship-Dataset - 2019, Wang et al.**](#SAR-Ship-Dataset)                            
* [**AIR-SARShip -1.0, 2.0 - 2019, Sun et al.**](#AIR-SARShip-1.0)                                       
* [**HRSID (High-Resolution SAR Images Dataset) - 2020, Wei et al.**](#HRSID) 
* [**LS-SSDD-v1.0 (Large-Scale SAR Ship Detection Dataset) - 2020, Zhang et al.**](#LS-SSDD-v1.0)
* [**SSDD (SAR Ship Detection Dataset) - 2021, Zhang et al.**](#SSDD1)
* [**DSSDD (Dual-polarimetric SAR Ship Detection Dataset) - 2021, Hu et al.**](#DSSDD)  
* [**SRSDD-v1.0 (SAR Rotation Ship Detection Dataset) - 2021, Lei et al.**](#SRSDD-v1.0) 
* [**xView3-SAR (Multi-modal SAR Ship Detection + Characterization Dataset) - 2022, Paolo, Lin, Gupta, et. al.**](#xView3-SAR)

## :satellite: Optical Satellite Datasets : 

- [**Airbus Aircraft Detection**](https://www.kaggle.com/airbusgeo/airbus-aircrafts-sample-dataset)
103 images of worlwide airports (Pleiades, 0.5m res., 2560px).
- [**Airbus Oil Storage Detection**](https://www.kaggle.com/airbusgeo/airbus-oil-storage-detection-dataset) 
Oil storage tank annotations, 98 worldwide images (SPOT, 1.2m res., 2560px).
- [**AFO - Aerial dataset of floating objects**](https://www.kaggle.com/jangsienicajzkowy/afo-aerial-dataset-of-floating-objects) 
3647 drone images from 50 scenes, 39991 objects with 6 categories.
- [**xView 2018 Detection Challenge**](http://xviewdataset.org)  
60 categories from helicopter to stadium, 1 million instances, Worldview-3 imagery (0.3m res.). 
- [**Open AI Challenge: Aerial Imagery of South Pacific Islands**](https://docs.google.com/document/d/16kKik2clGutKejU8uqZevNY6JALf4aVk2ELxLeR-msQ/edit) 
Tree position & 4 tree species, RGB UAV imagery (0.4m/0.8m res.), multiple AOIs in Tonga   
- [**NOAA Fisheries Steller Sea Lion Population Count**](https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count) 
5 sea lion categories, ~ 80k instances, ~ 1k aerial images.
- [**Stanford Drone Data**](http://cvgl.stanford.edu/projects/uav_data/?source=post_page---------------------------)   
60 aerial UAV videos over Stanford campus and bounding boxes, 6 classes.  
- [**HRSC2016 (High Resolution Ship Collection 2016) - 2016, Liu et al.**](#HRSC2016) 
- [**Airbus Ship Detection Challenge Dataset - 2018, Kaggle**](#Airbus) 
- [**xView Dataset  - 2018, Lam et al.**](#xView)  
- [**DOTA (Dataset for Object deTection in Aerial images) - 2018, Xia et al.**](#DOTA)
- [**TGRS-HRRSD (High-Resolution Remote Sensing object Detection) - 2018, Kaggle**](#TGRS)
- [**MASATI-v2 (MAritime SATellite Imagery dataset) - 2018,  Gallego et al.**](#MASATI)
- [**DIOR(object Detection In Optical Remote sensing images) - 2019, Li et al.**](#DIOR)    
- [**FGSD (Fine-Grained Ship Detection) - 2020, Chen et al.**](#FGSD) 
- [**PSDS (Peruvian Ship Data Set) + MSDS (Mini Ship Data Set) - 2020, Cordova et al.**](#PSDS)
- [**ShipRSImageNet - 2021, Zhang et al.**](#ShipRSImageNet)    
- [**S2-SHIPS - 2021, Ciocarlan et al.**](#S2ships)
- [**GF1-LRSD - 2021, Wu et al.**](#GF1LRSD) 
- [**VHRShips - 2022, Kizilkaya et al.**](#VHRShips)
- [**Cars Overhead With Context (COWC)**](https://gdo152.llnl.gov/cowc/) 
32k car bounding boxes, aerial imagery (0.15m res.).

## :satellite: Multi-sensor Satellite Datasets : 

- [**xView3 Dark Vessel Detection 2021**](https://iuu.xview.us/) 
Maritime object bounding boxes for 1k Sentinel-1 scenes (VH & VV polarizations), ancillary data (land/ice mask, bathymetry, wind speed, direction, quality).
- [**NEON Tree Crowns Dataset**](https://zenodo.org/record/3765872#.YHs-MBMzbUI)    
Individual tree crown objects, height&area estimates, 100 million instances, 37 geographic sites across the US, [DeepForest](https://github.com/weecology/DeepForest), Has LiDAR data. 
- [**NIST DSE Plant Identification with NEON Remote Sensing Data**](https://www.ecodse.org) 
Tree position, tree species and crown parameters, hyperspectral (1m res.) & RGB imagery (0.25m res.), LiDAR point cloud and canopy height model   

  

# 5. Semantic Segmentation

## :satellite: Radar Satellite Datasets : 

* [**SSDD (SAR Ship Detection Dataset) - 2017, Li et al.**](#SSDD) 
 
## :satellite: Optical Satellite Datasets : 

- [**FloodNet**](https://github.com/BinaLab/FloodNet-Supervised_v1.0)  
  2343 image chips (drone imagery), 10 landcover categories.
- [**LoveDA**](https://github.com/Junjue-Wang/LoveDA)  
5987 image chips (Google Earth), 7 landcover categories.
- [**FloodNet Challenge**](http://www.classic.grss-ieee.org/earthvision2021/challenge.html) 
2343 UAV images, 2 competition tracks (Binary & semantic flood classification; Object counting & condition recognition)  
- [**Dynamic EarthNet Challenge**](http://www.classic.grss-ieee.org/earthvision2021/challenge.html) 
Weekly Planetscope time-series (3m res.) over 2 years, 75 aois, landcover labels (7 categories), 2 competition tracks (Binary land cover classification & multi-class change detection)  
- [**Sentinel-2 Cloud Mask Catalogue**](https://zenodo.org/record/4172871) 
513 cropped subscenes (1022x1022 pixels) taken randomly from entire 2018 Sentinel-2 archive. 
- [**MiniFrance**](https://ieee-dataport.org/open-access/minifrance)
  2000 very high resolution aerial images over 16 cities in France (50cm res., from IGN BDORTHO).
- [**LandCoverNet: A Global Land Cover Classification Training Dataset**](https://doi.org/10.34911/rdnt.d2ce8i) 
1980 image chips of 256 x 256 pixels in V1.0 spanning 66 tiles of Sentinel-2. 
- [**LandCover.ai: Dataset for Automatic Mapping of Buildings, Woodlands and Water from Aerial Imagery**](http://landcover.ai/) 
41 orthophotos (9000x9000 px) over Poland, Aerial Imagery (25cm & 50cm res.)
 - [**95-Cloud: A Cloud Segmentation Dataset**](https://github.com/SorourMo/95-Cloud-An-Extension-to-38-Cloud-Dataset)  
34701 manually segmented 384x384 patches with cloud masks, Landsat 8 imagery (R,G,B,NIR; 30 m res.)
- [**SkyScapes: Urban infrastructure & lane markings**](https://www.dlr.de/eoc/en/desktopdefault.aspx/tabid-12760/22294_read-58694/) 
Highly accurate street lane markings & urban infrastructure . Aerial imagery (0.13 m res.).
- [**Open AI Challenge: Caribbean**](https://www.drivendata.org/competitions/58/disaster-response-roof-type/page/143/)
Predict building roof type (22,553), RGB UAV imagery (4cm res.).
- [**ALCD Reference Cloud Masks**](https://zenodo.org/record/1460961#.XYCTRzYzaHt) 
8 classes (inc. cloud and cloud shadow) for 38 Sentinel-2 scenes (10 m res.).
 - [**Agricultural Crop Cover Classification Challenge**](https://crowdanalytix.com/contests/agricultural-crop-cover-classification-challenge) 
2 main categories corn and soybeans, Landsat 8 imagery (30m res.).
 - [**RoadNet**](https://github.com/yhlleo/RoadNet) 
Road network labels, high-res Google Earth imagery, 21 regions.
- [**SpaceNet 3: Road Network Detection**](https://spacenet.ai/spacenet-roads-dataset/) 
8000 km of roads in 5 city aois, 3/8band Worldview-3 imagery (0.3m res.). 
- [**DSTL Satellite Imagery Feature Detection Challenge**](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection) 
10 land cover categories from crops to vehicle small, 57 1x1km images, 3/16-band Worldview 3 imagery (0.3m-7.5m res.).
- [**SPARCS: S2 Cloud Validation data**](https://www.usgs.gov/land-resources/nli/landsat/spatial-procedures-automated-removal-cloud-and-shadow-sparcs-validation) 
7 categories, 80 1kx1k px. subset Landsat 8 scenes (30m res.).
 - [**Inria Aerial Image Labeling**](https://project.inria.fr/aerialimagelabeling/contest/) 
Building footprint masks, RGB aerial imagery (0.3m res.).
 
 
## :satellite: Multi-sensor Satellite Datasets : 
 - [**Open Cities AI Challenge**](https://www.drivendata.org/competitions/60/building-segmentation-disaster-resilience/page/150/)  
790k building footprints from Openstreetmap (2 label quality categories), aerial imagery (0.03-0.2m resolution, RGB, 11k 1024x1024 chips, COG format).
- [**DroneDeploy Segmentation Dataset**](https://github.com/dronedeploy/dd-ml-segmentation-benchmark) 
Drone imagery (0.1m res., RGB), labels & elevation data, baseline model implementation.
- [**SpaceNet 5: Automated Road Network Extraction & Route Travel Time Estimation**](https://spacenet.ai/sn5-challenge/) 
2300 image chips, street geometries with location, shape and estimated travel time, 3/8band Worldview-3 imagery (0.3m res.).
- [**SEN12MS**](https://mediatum.ub.tum.de/1474000)
180,748 corresponding image triplets containing Sentinel-1 (VV&VH), Sentinel-2 (all bands, cloud-free), and MODIS-derived land cover maps (IGBP, LCCS, 17 classes, 500m res.). 
- [**Slovenia Land Cover Classification**](http://eo-learn.sentinel-hub.com)
10 land cover classes, temporal stack of hyperspectral Sentinel-2 imagery.
- [**Urban 3D Challenge**](https://spacenet.ai/the-ussocom-urban-3d-competition/)  
157k building footprint masks, RGB orthophotos (0.5m res.), DSM/DTM.
- [**ISPRS Potsdam 2D Semantic Labeling Contest**](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html) 
6 urban land cover classes, raster mask labels, 4-band RGB-IR aerial imagery (0.05m res.) & DSM, 38 image patches    
  



## 6. Other Focus / Multiple Tasks

- [**SEN12MS-CR**](https://patricktum.github.io/cloud_removal/) & [**SEN12MS-CR-TS**](https://patricktum.github.io/cloud_removal/)   
A multi-modal and mono-temporal data set for cloud removal. Sentinel-1 & Sentinel-2, 2018. 175 globally distributed aois.
- [**IEEE Data Fusion Contest 2022**](https://www.grss-ieee.org/community/technical-committees/2022-ieee-grss-data-fusion-contest/)
Semi-supervised semantic segmentation, 19 cities and surroundings with multi-sensor tiles (VHR Aerial imagery 50cm res., Elevation model) & per pixel labels (contains landcover / landuse classes from UrbanAtlas 2012), 
- [**IEEE Data Fusion Contest 2021**](https://www.grss-ieee.org/community/technical-committees/2021-ieee-grss-data-fusion-contest-track-dse/) 
Detection of settlements without electricity, 98 multi-temporal/multi-sensor tiles ( Sentinel-1, Sentinel-2, Landsat-8, VIIRS), per chip & per pixel labels (contains buildings, presence electricity). 
- [**University-1652: Drone-based Geolocalization (Image Retrieval)**](https://github.com/layumi/University1652-Baseline)   
Corresponding imagery from drone, satellite and ground camera of 1,652 university buildings.   
- [**IEEE Data Fusion Contest 2020**](https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest) 
Land cover classification based on SEN12MS dataset (see category Semantic Segmentation on this list), low- and high-resolution tracks.
- [**IEEE Data Fusion Contest 2019**](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019) 
Multiple tracks: Semantic 3D reconstruction, Semantic Stereo, 3D-Point Cloud Classification. Worldview-3 (8-band, 0.35cm res.) satellite imagery, LiDAR (0.80m pulse spacing, ASCII format), semantic labels, urban setting USA, baseline methods provided,
- [**IEEE Data Fusion Contest 2018**](https://ieee-dataport.org/open-access/2018-ieee-grss-data-fusion-challenge-%E2%80%93-fusion-multispectral-lidar-and-hyperspectral-data) 
20 land cover categories by fusing three data sources: Multispectral LiDAR, Hyperspectral (1m), RGB imagery (0.05m res.).
- [**DEEPGLOBE - 2018 Satellite Challange**](http://deepglobe.org/index.html)
Three challenge tracks: Road Extraction, Building Detection, Land cover classification. 
- [**TiSeLaC: Time Series Land Cover Classification Challenge**](https://sites.google.com/site/dinoienco/tiselac-time-series-land-cover-classification-challenge?authuser=0) 
Land cover time series classification (9 categories), Landsat-8 (23 images time series, 10 band features, 30m res.).
- [**Multi-View Stereo 3D Mapping Challenge**](https://www.iarpa.gov/challenges/3dchallenge.html) 
Develop a Multi-View Stereo (MVS) 3D mapping algorithm that can convert high-resolution Worldview-3 satellite images to 3D point clouds, 0.2m lidar ground truth data.   
- [**Draper Satellite Image Chronology**](https://www.kaggle.com/c/draper-satellite-image-chronology) 
Predict the chronological order of images taken at the same locations over 5 days, Kaggle kernels


</div align="justify">    


[Go To TOP](#TOP)
 
