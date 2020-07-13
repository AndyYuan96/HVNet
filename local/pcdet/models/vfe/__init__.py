from .vfe_utils import MeanVoxelFeatureExtractor, PillarFeatureNetOld2, MVFFeatureNet,MVFFeatureNetDVP, HVFeatureNet, HVFeatureNetPaper, HVFeatureNetFinal


vfe_modules = {
    'MeanVoxelFeatureExtractor': MeanVoxelFeatureExtractor,
    'PillarFeatureNetOld2': PillarFeatureNetOld2,
    'MVFFeatureNet': MVFFeatureNet,
    'MVFFeatureNetDVP' : MVFFeatureNetDVP,
    'HVFeatureNet': HVFeatureNet,
    'HVFeatureNetPaper': HVFeatureNetPaper,
    'HVFeatureNetFinal': HVFeatureNetFinal
}