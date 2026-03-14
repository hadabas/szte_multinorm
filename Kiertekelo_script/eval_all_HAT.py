import argparse
import datetime
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

import robustbench as rb
import autoattack

import data
import utils
import other_utils

from model_zoo.fast_models import PreActResNet18
from model_zoo.wide_resnet import WideResNet
from spgd_l0.autoattack.saa import SparseAutoAttack

from border_attack.borderinner import BorderInnerAttack, BorderInnerAttack_targeted
from border_attack.borderattack import BorderAttack, BorderAttack_targeted

from hat_models.preact_resnet import preact_resnet as hat_load_preact_resnet


eps_dict = {
    "cifar10":  {"Linf": 8.0 / 255.0, "L2": 0.5, "L1": 12.0},
    "cifar100": {"Linf": 8.0 / 255.0, "L2": 0.5, "L1": 12.0},
    "imagenet": {"Linf": 4.0 / 255.0, "L2": 2.0, "L1": 255.0},
}


# -----------------------------
# AutoAttack evaluation helpers
# -----------------------------
def eval_single_norm(model, x, y, norm="Linf", eps=8.0 / 255.0, bs=1000, log_path=None):
    adversary = autoattack.AutoAttack(model, norm=norm, eps=eps, log_path=log_path)
    adversary.attacks_to_run = ["apgd-ce", "apgd-t"]
    with torch.no_grad():
        x_adv = adversary.run_standard_evaluation(x, y, bs=bs)
    acc = rb.utils.clean_accuracy(model, x_adv, y, device="cuda")
    other_utils.check_imgs(x_adv, x, norm)
    print(f"[AA {norm}] robust accuracy: {acc:.2%}")
    return x_adv


def eval_norms_union(model, x, y, l_norms, l_epss, bs=1000, log_path=None, n_cls=10):
    """
    Returns:
      clean_mask: bool[N] correct on clean
      union_mask: bool[N] correct on clean AND on all provided norms
    """
    l_x_adv_cpu = []
    logger = other_utils.Logger(log_path)

    for norm, eps in zip(l_norms, l_epss):
        x_adv = eval_single_norm(model, x, y, norm=norm, eps=eps, bs=bs, log_path=log_path)
        l_x_adv_cpu.append(x_adv.detach().cpu())
        del x_adv
        torch.cuda.empty_cache()

    # clean mask
    _, output = utils.get_accuracy_and_logits(model, x, y, batch_size=bs, n_classes=n_cls)
    clean_mask = output.to(y.device).argmax(1).eq(y).detach().cpu()

    clean_acc = clean_mask.float().mean().item() * 100.0
    print(f"[clean] accuracy: {clean_acc:.2f}%")
    logger.log("")
    logger.log(f"clean accuracy: {clean_mask.float().mean():.1%}")

    # union across norms
    union_m = clean_mask.clone()
    for norm, x_adv_cpu in zip(l_norms, l_epss,):  # placeholder to match loop below
        pass

    for norm, x_adv_cpu in zip(l_norms, l_x_adv_cpu):
        _, out_adv = utils.get_accuracy_and_logits(model, x_adv_cpu, y, batch_size=bs, n_classes=n_cls)
        m = out_adv.to(y.device).argmax(1).eq(y).detach().cpu()
        acc = m.float().mean().item() * 100.0
        print(f"[AA {norm}] (recheck) robust: {acc:.2f}%")
        logger.log(f"robust accuracy {norm}: {m.float().mean():.1%}")
        union_m &= m

    union_acc = union_m.float().mean().item() * 100.0
    print(f"[AA union {'+'.join(l_norms)}] robust accuracy: {union_acc:.2f}%")
    logger.log(f"robust accuracy {'+'.join(l_norms)}: {union_m.float().mean():.1%}")

    return clean_mask, union_m


# -----------------------------
# Mask utilities
# -----------------------------
@torch.no_grad()
def mask_correct(model, x, y, bs=512, n_cls=10):
    _, logits = utils.get_accuracy_and_logits(model, x, y, batch_size=bs, n_classes=n_cls)
    return logits.to(y.device).argmax(1).eq(y).detach().cpu()


def run_attack_full_batched(
    name,
    attacker_ctor,
    attacker_kwargs,
    model,
    x_cpu,
    y_cpu,
    *,
    bs_attack,
    bs_eval,
    n_cls,
    device="cuda",
):
    """
    Runs attacker on the FULL dataset in batches of size bs_attack.
    Returns full-length correctness mask after attack (so robust acc = mask.mean()).
    """
    n = y_cpu.numel()
    out_mask = torch.zeros(n, dtype=torch.bool)

    print(f"[{name}] running on FULL set (N={n}) | bs_attack={bs_attack}")
    for start in range(0, n, bs_attack):
        end = min(start + bs_attack, n)
        x_b = x_cpu[start:end].to(device, non_blocking=True)
        y_b = y_cpu[start:end].to(device, non_blocking=True)

        attacker = attacker_ctor(**attacker_kwargs)
        x_adv = attacker.perturb(x_b, y_b)

        out_mask[start:end] = mask_correct(
            model, x_adv, y_b, bs=min(bs_eval, end - start), n_cls=n_cls
        )

        del attacker, x_adv, x_b, y_b
        torch.cuda.empty_cache()

        if (start // bs_attack) % 50 == 0:
            done = end
            acc_so_far = out_mask[:done].float().mean().item() * 100.0
            print(f"[{name}] progress {done}/{n} | robust-so-far: {acc_so_far:.2f}%")

    acc = out_mask.float().mean().item() * 100.0
    print(f"[{name}] robust accuracy (FULL set): {acc:.2f}%")
    return out_mask


def union_mask(*masks):
    m = masks[0].clone()
    for mm in masks[1:]:
        m &= mm
    return m


def print_union(title, *masks):
    acc = union_mask(*masks).float().mean().item() * 100.0
    print(f"[UNION] {title}: {acc:.2f}%")
    return acc


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_name", type=str, default="./trained_models/RAMP_2PAIR_DUAL2L1_L12L0/ep_80_0.pth")
    p.add_argument("--n_ex", type=int, default=10000)
    p.add_argument("--batch_size_eval", type=int, default=100)
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "imagenet"])
    p.add_argument("--only_clean", action="store_true")

    p.add_argument("--l_norms", type=str, default="Linf L2 L1")
    p.add_argument("--l_eps", type=str, default=None)

    # optional attacks
    p.add_argument("--run_border", action="store_true")
    p.add_argument("--run_border_inner", action="store_true")

    # BorderAttack config
    p.add_argument("--border_eps", type=float, default=1.0)
    p.add_argument("--border_n_iter_u", type=int, default=100)
    p.add_argument("--border_restarts_u", type=int, default=5)
    p.add_argument("--border_n_iter_t", type=int, default=100)
    p.add_argument("--border_restarts_t", type=int, default=1)
    p.add_argument("--border_n_target_classes", type=int, default=9)

    # BorderInner config
    p.add_argument("--bi_eps_border", type=float, default=1.0)
    p.add_argument("--bi_eps_inner", type=float, default=8.0 / 255.0)
    p.add_argument("--bi_n_iter_u", type=int, default=100)
    p.add_argument("--bi_restarts_u", type=int, default=5)
    p.add_argument("--bi_loss_u", type=str, default="combined")
    p.add_argument("--bi_n_iter_t", type=int, default=100)
    p.add_argument("--bi_restarts_t", type=int, default=1)
    p.add_argument("--bi_n_target_classes", type=int, default=9)

    # L0 SparseAutoAttack
    p.add_argument("--l0_eps", type=float, default=1.0)
    p.add_argument("--l0_k", type=int, default=20)
    p.add_argument("--l0_alpha", type=float, default=0.25)
    p.add_argument("--l0_beta", type=float, default=0.25)
    p.add_argument("--l0_n_iters", type=int, default=300)
    p.add_argument("--l0_bs", type=int, default=512)
    p.add_argument("--l0_black_iters", type=int, default=10000)
    p.add_argument("--l0_max_candidates", type=int, default=5)

    return p.parse_args()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    args = parse_args()

    # ---- load data ----
    if args.dataset == "cifar10":
        x_test, y_test = data.load_cifar10(args.n_ex, data_dir=args.data_dir, device="cpu")
        n_cls = 10
    elif args.dataset == "cifar100":
        if hasattr(data, "load_cifar100"):
            x_test, y_test = data.load_cifar100(args.n_ex, data_dir=args.data_dir, device="cpu")
        else:
            x_test, y_test = rb.data.load_cifar100(n_examples=args.n_ex, data_dir=args.data_dir)
        n_cls = 100
    elif args.dataset == "imagenet":
        x_test, y_test = data.load_imagenet(args.n_ex, device="cpu")
        n_cls = 1000
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # ---- load model ---- !!ÁLTALÁBAN EZT KELL ÁLLÍTANI!!!
    if args.dataset == "cifar10":
        model = hat_load_preact_resnet("preact-resnet18",10).cuda()
    elif args.dataset == "cifar100":
        model = hat_load_preact_resnet("preact-resnet18",100).cuda()
    else:
        model = hat_load_preact_resnet("preact-resnet18",1000).cuda()

    checkpoint = torch.load(args.model_name)
    state_dict = checkpoint["model_state_dict"]
    state_dict = {k.replace("module.0.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    del checkpoint

    # ---- clean accuracy ----
    clean_acc = rb.utils.clean_accuracy(model, x_test, y_test, device="cuda")
    print(f"[clean] accuracy (rb): {clean_acc*100:.2f}%")
    if args.only_clean:
        sys.exit(0)

    # ---- norms / eps ----
    l_norms = args.l_norms.split(" ")
    if args.l_eps is None:
        l_epss = [eps_dict[args.dataset][c] for c in l_norms]
    else:
        norm, size = args.l_eps.split("_")[0], float(args.l_eps.split("_")[1])
        eps_dict[args.dataset][norm] = size
        l_epss = [eps_dict[args.dataset][c] for c in l_norms]

    # ---- AutoAttack base unions ----
    clean_mask, aa_union_3 = eval_norms_union(
        model, x_test, y_test,
        l_norms=l_norms, l_epss=l_epss,
        bs=args.batch_size_eval,
        log_path=None,
        n_cls=n_cls
    )

    masks = {
        "clean": clean_mask,
        "AA3": aa_union_3,
    }

    # ---- BorderAttack (U+T combined), robust on FULL test set ----
    if args.run_border:
        border_u = run_attack_full_batched(
            name="BORDER-U",
            attacker_ctor=BorderAttack,
            attacker_kwargs=dict(
                predict=model,
                n_iter=args.border_n_iter_u,
                norm="Linf",
                n_restarts=args.border_restarts_u,
                eps=args.border_eps,
                loss="combined",
                verbose=False,
                device=torch.device("cuda"),
            ),
            model=model,
            x_cpu=x_test,
            y_cpu=y_test,
            bs_attack=args.batch_size_eval,
            bs_eval=args.batch_size_eval,
            n_cls=n_cls,
            device="cuda",
        )

        border_t = run_attack_full_batched(
            name="BORDER-T",
            attacker_ctor=BorderAttack_targeted,
            attacker_kwargs=dict(
                predict=model,
                n_iter=args.border_n_iter_t,
                norm="Linf",
                n_restarts=args.border_restarts_t,
                eps=args.border_eps,
                n_target_classes=args.border_n_target_classes,
                verbose=False,
                device=torch.device("cuda"),
            ),
            model=model,
            x_cpu=x_test,
            y_cpu=y_test,
            bs_attack=args.batch_size_eval,
            bs_eval=args.batch_size_eval,
            n_cls=n_cls,
            device="cuda",
        )

        masks["BORDER"] = union_mask(border_u, border_t)
        border_acc = masks["BORDER"].float().mean().item() * 100.0
        print(f"[BORDER] robust accuracy (U+T combined, FULL set): {border_acc:.2f}%")

    else:
        print("[BORDER] skipped (use --run_border)")

    # ---- BorderInnerAttack (U+T combined), robust on FULL test set ----
    if args.run_border_inner:
        bi_u = run_attack_full_batched(
            name="BORDER-INNER-U",
            attacker_ctor=BorderInnerAttack,
            attacker_kwargs=dict(
                predict=model,
                n_iter=args.bi_n_iter_u,
                norm="Linf",
                n_restarts=args.bi_restarts_u,
                eps=args.bi_eps_border,
                eps_inner=args.bi_eps_inner,
                loss=args.bi_loss_u,
                verbose=False,
                device=torch.device("cuda"),
            ),
            model=model,
            x_cpu=x_test,
            y_cpu=y_test,
            bs_attack=args.batch_size_eval,
            bs_eval=args.batch_size_eval,
            n_cls=n_cls,
            device="cuda",
        )

        bi_t = run_attack_full_batched(
            name="BORDER-INNER-T",
            attacker_ctor=BorderInnerAttack_targeted,
            attacker_kwargs=dict(
                predict=model,
                n_iter=args.bi_n_iter_t,
                norm="Linf",
                n_restarts=args.bi_restarts_t,
                eps=args.bi_eps_border,
                eps_inner=args.bi_eps_inner,
                n_target_classes=args.bi_n_target_classes,
                verbose=False,
                device=torch.device("cuda"),
            ),
            model=model,
            x_cpu=x_test,
            y_cpu=y_test,
            bs_attack=args.batch_size_eval,
            bs_eval=args.batch_size_eval,
            n_cls=n_cls,
            device="cuda",
        )

        masks["BI"] = union_mask(bi_u, bi_t)
        bi_acc = masks["BI"].float().mean().item() * 100.0
        print(f"[BORDER-INNER] robust accuracy (U+T combined, FULL set): {bi_acc:.2f}%")

    else:
        print("[BORDER-INNER] skipped (use --run_border_inner)")

    # ---- L0 (SparseAutoAttack) ----
    spgd_args = argparse.Namespace(
        exp="test",
        dataset=args.dataset,
        data_dir=args.data_dir,
        ckpt="not-used-here",
        eps=args.l0_eps,
        k=args.l0_k,
        alpha=args.l0_alpha,
        beta=args.l0_beta,
        n_iters=args.l0_n_iters,
        bs=args.l0_bs,
        n_examples=args.n_ex,
        model="ramp",
        unprojected_gradient=False,
        gpu=0,
    )

    if spgd_args.dataset == "cifar10":
        x_rb, y_rb = rb.data.load_cifar10(n_examples=spgd_args.n_examples, data_dir=spgd_args.data_dir)
        max_candidates = args.l0_max_candidates
    elif spgd_args.dataset == "cifar100":
        x_rb, y_rb = rb.data.load_cifar100(n_examples=spgd_args.n_examples, data_dir=spgd_args.data_dir)
        max_candidates = args.l0_max_candidates
    elif spgd_args.dataset == "imagenet":
        x_rb, y_rb = rb.data.load_imagenet(n_examples=spgd_args.n_examples, data_dir=spgd_args.data_dir)
        max_candidates = 10
    else:
        raise ValueError(f"Unknown dataset for rb loader: {spgd_args.dataset}")

    idxs = torch.arange(len(y_rb))
    dataset = TensorDataset(x_rb, y_rb, idxs)
    loader = DataLoader(dataset, batch_size=spgd_args.bs, shuffle=False, num_workers=2)

    print("-- Starting SPGD_L0 evaluation process --")
    attacker = SparseAutoAttack(
        model,
        spgd_args,
        black_iters=args.l0_black_iters,
        max_candidates=max_candidates,
    )
    _, _, spgd_robust_mask, time_used = attacker(loader)
    masks["L0"] = spgd_robust_mask.detach().cpu()
    l0_acc = masks["L0"].float().mean().item() * 100.0
    print(f"[L0] robust accuracy (FULL set): {l0_acc:.2f}%")

    # -----------------------------
    # Union summary (combined Border/BI only)
    # -----------------------------
    print("\n================ UNION SUMMARY ================")

    print_union("L1+L2+Linf", masks["AA3"])
    print_union("L0+L1+L2+Linf", masks["AA3"], masks["L0"])

    if "BORDER" in masks:
        print_union("L0+L1+L2+Linf+Border", masks["AA3"], masks["L0"], masks["BORDER"])
    else:
        print("[UNION] L0+L1+L2+Linf+Border: (skipped; run with --run_border)")

    if "BORDER" in masks and "BI" in masks:
        print_union("L0+L1+L2+Linf+Border+BorderInner", masks["AA3"], masks["L0"], masks["BORDER"], masks["BI"])
    elif "BI" in masks:
        print_union("L0+L1+L2+Linf+BorderInner", masks["AA3"], masks["L0"], masks["BI"])
    else:
        print("[UNION] ...+BorderInner: (skipped; run with --run_border_inner)")

    all_masks = [masks["AA3"], masks["L0"]]
    label = "L0+L1+L2+Linf"
    if "BORDER" in masks:
        all_masks.append(masks["BORDER"])
        label += "+Border"
    if "BI" in masks:
        all_masks.append(masks["BI"])
        label += "+BorderInner"
    print_union(f"FULL ({label})", *all_masks)

    print("\nTime used (L0 attacker):", datetime.timedelta(seconds=time_used))