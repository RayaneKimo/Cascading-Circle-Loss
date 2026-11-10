from cascl.losses.cascl import CascadingCircleLoss
from cascl.losses.cascl_supcon import CascadingCircleLoss as CascadingCircleLossSupCon
from cascl.losses.cascl_rectified import CascadingCircleLoss as CascadingCircleLossRectified
from cascl.losses.cascl_two_regime import CascadingCircleLoss as CascadingCircleLossTwoRegime
from cascl.losses.cascl_neg_relative import CascadingCircleLoss as CascadingCircleLossNegRelative
from cascl.losses.baselines import (
    InfoNCELoss,
    CircleLoss,
    TripletLoss,
    MultipleNegativesRankingLoss,
    MarginMSELoss,
    RINCELoss,
    CLOCLoss,
)
