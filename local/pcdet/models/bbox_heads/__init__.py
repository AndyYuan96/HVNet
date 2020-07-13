from .rpn_head import RPNV2,HVHead,FPNHead,RPNV3,RPNV4


bbox_head_modules = {
    'RPNV2': RPNV2,
    'HVHead': HVHead,
    'FPNHead': FPNHead,
    'RPNV3': RPNV3,
    'RPNV4': RPNV4
}