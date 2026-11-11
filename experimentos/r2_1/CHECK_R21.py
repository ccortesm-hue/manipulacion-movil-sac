"""One physical triple (baseline/OFF/ON); infrastructure check, NOT validation."""
import argparse
from pathlib import Path
from TRAIN_R21 import parser,TrainingEnv,validate_source,build,load_norm,torch
from EVALUATE_R21 import evaluate,make_cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--probe',action='store_true',help='Artificial nonzero bias solely to test physical wiring; no trained candidate')
    args=p.parse_args()
    cfg=parser().parse_args(['--output',str(args.output),'--device','cpu','--buffer-size','6000'])
    validate_source(cfg.source);torch.set_num_threads(1)
    norm=load_norm(cfg.source.with_name('vecnormalize.pkl'))
    env=TrainingEnv(norm,cfg.seed)
    try:
        model,_=build(env,norm,cfg)
        if args.probe:
            with torch.no_grad():
                model.actor.mu.bias[5]=.3
                model.actor.mu.bias[6]=-.2
        report=evaluate(model,norm,make_cases([1],[1.],1,401000000),args.output,
                        lambda stage,**kw:print(stage,kw,flush=True))
        if args.probe:
            assert report['residual_physically_active']
            print('SMOKE_OK_ARTIFICIAL_PROBE_NOT_A_TRAINED_MODEL',report,flush=True)
        else:
            assert not report['residual_physically_active']
            assert not report['qualified']
            print('SMOKE_OK_INITIAL_ZERO_NOT_PROMOTED',report,flush=True)
    finally: env.close();norm.close()


if __name__=='__main__': main()
