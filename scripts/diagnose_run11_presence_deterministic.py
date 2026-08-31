#!/usr/bin/env python3
"""Deterministic CPU-only polynomial presence diagnosis for the Run11 checkpoint.

Uses only official DeepMind generators. Every bank explicitly reseeds all runtime
RNGs. Threshold is chosen on calibration banks and evaluated on disjoint validation
banks. No model training occurs.
"""

import argparse
import copy
import importlib.util
import json
import pathlib

import diagnose_run11_presence as common


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--repo-root',type=pathlib.Path,required=True)
    p.add_argument('--checkpoint',type=pathlib.Path,required=True)
    p.add_argument('--run11-audit',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True)
    return p.parse_args()


def seeded_bank(audit, ns, base_spec, adapter_seed, global_seed, count):
    spec=copy.deepcopy(base_spec)
    spec['seed']=int(adapter_seed)
    spec['count']=int(count)
    common.reseed_official(ns,int(global_seed))
    return audit.build_official_bank(ns,spec), spec


def equation_signature(examples):
    return [e['eq'] for e in examples]


def assert_reseed_reproducible(audit,ns,base_spec,adapter_seed,global_seed):
    a,_=seeded_bank(audit,ns,base_spec,adapter_seed,global_seed,32)
    b,_=seeded_bank(audit,ns,base_spec,adapter_seed,global_seed,32)
    if equation_signature(a)!=equation_signature(b):
        raise RuntimeError(f"FULL_RNG_RESEED_NOT_REPRODUCIBLE {base_spec['split']}/{base_spec['module']}")
    print(f"RESEED_REPRODUCIBLE {base_spec['split']}/{base_spec['module']} examples=32")


def main():
    args=parse_args()
    root=args.repo_root.resolve(); checkpoint=args.checkpoint.resolve(); output=args.output.resolve()
    run11=json.loads(args.run11_audit.resolve().read_text())

    audit_path=root/'colab/generalization_audit.py'
    s=importlib.util.spec_from_file_location('dm_family_audit',audit_path)
    audit=importlib.util.module_from_spec(s); s.loader.exec_module(audit)
    ns=audit.load_runtime(root)
    torch=ns['torch']
    if torch.cuda.is_available() or str(ns['device'])!='cpu':
        raise RuntimeError('CPU_ONLY_DIAGNOSIS_REFUSES_CUDA')
    ns['load_mai5'](str(checkpoint))
    print(f"Loaded Run11 checkpoint adam_step={ns['adam_step']} device={ns['device']}")

    i_base=audit.BANK_SPECS['interpolate_polynomial']
    e_base=audit.BANK_SPECS['extrapolate_polynomial']

    # Prove that explicit reseeding really makes the official generator stable in
    # this runtime before using any result.
    assert_reseed_reproducible(audit,ns,i_base,0xB0010001,0x51010001)
    assert_reseed_reproducible(audit,ns,e_base,0xB0010002,0x51010002)

    # Threshold tuning banks.
    cal_i,cal_i_spec=seeded_bank(audit,ns,i_base,0xC411B001,0x61010001,128)
    cal_e,cal_e_spec=seeded_bank(audit,ns,e_base,0xC411B002,0x61010002,128)
    # Disjoint validation banks, never used to choose the threshold.
    val_i,val_i_spec=seeded_bank(audit,ns,i_base,0xD411B001,0x71010001,256)
    val_e,val_e_spec=seeded_bank(audit,ns,e_base,0xD411B002,0x71010002,256)

    cal_i_rows=common.collect_bank(ns,cal_i); cal_e_rows=common.collect_bank(ns,cal_e)
    val_i_rows=common.collect_bank(ns,val_i); val_e_rows=common.collect_bank(ns,val_e)
    thresholds=[x/100.0 for x in range(5,96)]

    baseline={'interpolate':common.metrics(val_i_rows,0.5),'extrapolate':common.metrics(val_e_rows,0.5)}
    chosen=common.choose_threshold(cal_i_rows,cal_e_rows,thresholds)
    _,chosen_t,cal_mi,cal_me=chosen
    selected={'interpolate':common.metrics(val_i_rows,chosen_t),'extrapolate':common.metrics(val_e_rows,chosen_t)}
    oracle={'interpolate':common.best_for_bank(val_i_rows,thresholds),'extrapolate':common.best_for_bank(val_e_rows,thresholds)}

    i_count=float(i_base['min_count_accuracy']); e_count=float(e_base['min_count_accuracy'])
    i_within=float(i_base['min_within_one']); e_within=float(e_base['min_within_one'])
    selected_count_pass=selected['interpolate']['root_count_accuracy']>=i_count and selected['extrapolate']['root_count_accuracy']>=e_count
    oracle_count_pass=oracle['interpolate']['root_count_accuracy']>=i_count and oracle['extrapolate']['root_count_accuracy']>=e_count
    selected_accuracy_floor_pass=(selected_count_pass and selected['interpolate']['within_one_ratio']>=i_within and selected['extrapolate']['within_one_ratio']>=e_within)

    run11_ref={
        'interpolate':run11['banks']['interpolate_polynomial']['trained'],
        'extrapolate':run11['banks']['extrapolate_polynomial']['trained'],
    }
    payload={
        'schema':'RUN11_POLYNOMIAL_PRESENCE_DETERMINISTIC_V1',
        'cpu_only':True,
        'training_performed':False,
        'project_synthetic_examples':0,
        'checkpoint':checkpoint.name,
        'adam_step':int(ns['adam_step']),
        'generator_contract':'official DeepMind only; explicit Python+NumPy+Torch reseed per bank; 32-example repeat probe passed',
        'run11_original_audit_reference':{
            k:{'root_count_accuracy':v['root_count_accuracy'],'within_one_ratio':v['within_one_ratio'],'rmse':v['rmse'],'mae':v['mae']} for k,v in run11_ref.items()
        },
        'banks':{
            'calibration_interpolate':audit.bank_metadata(cal_i_spec,cal_i),
            'calibration_extrapolate':audit.bank_metadata(cal_e_spec,cal_e),
            'validation_interpolate':audit.bank_metadata(val_i_spec,val_i),
            'validation_extrapolate':audit.bank_metadata(val_e_spec,val_e),
        },
        'gate_accuracy_floors':{
            'interpolate':{'root_count':i_count,'within_one':i_within},
            'extrapolate':{'root_count':e_count,'within_one':e_within},
        },
        'validation_at_threshold_0_5':baseline,
        'selected_threshold':float(chosen_t),
        'selected_on_calibration':{'interpolate':cal_mi,'extrapolate':cal_me},
        'selected_threshold_on_disjoint_validation':selected,
        'oracle_best_threshold_on_validation_diagnostic_only':oracle,
        'threshold_only_can_pass_count_gate':bool(selected_count_pass),
        'oracle_threshold_can_pass_both_count_gates':bool(oracle_count_pass),
        'selected_threshold_can_pass_count_and_within1_floors':bool(selected_accuracy_floor_pass),
        'presence_shape_validation_interpolate':common.presence_shape(val_i_rows),
        'presence_shape_validation_extrapolate':common.presence_shape(val_e_rows),
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    print('VALIDATION_0.50',json.dumps(baseline,sort_keys=True))
    print(f'SELECTED_THRESHOLD={chosen_t:.2f}')
    print('SELECTED_VALIDATION',json.dumps(selected,sort_keys=True))
    print('ORACLE_VALIDATION',json.dumps(oracle,sort_keys=True))
    print(f'THRESHOLD_ONLY_CAN_PASS_COUNT_GATE={selected_count_pass}')
    print(f'ORACLE_THRESHOLD_CAN_PASS_BOTH_COUNT_GATES={oracle_count_pass}')
    print(f'SELECTED_THRESHOLD_CAN_PASS_COUNT_AND_WITHIN1_FLOORS={selected_accuracy_floor_pass}')


if __name__=='__main__':
    main()
