# from .runner import run_net
from .runner_test_cls import test_net as test_cls
from .runner_test_partseg import test_net as test_partseg
from .runner_test_semseg import test_net as test_semseg

from .runner_finetune_cls import run_net as finetune_cls
from .runner_finetune_partseg import run_net as finetune_partseg
from .runner_finetune_semseg import run_net as finetune_semseg

from .runner_svm import run_net_svm as svm
from .runner_vis import test_net as visualiztion
from .runner_tsne import tsne_net as tsne

from .runner_pretrain import run_net as pretrain





