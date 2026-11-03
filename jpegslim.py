#!/usr/bin/env python3
"""
JPEGSlim - otimizador de JPEG com qualidade perceptual controlada.

Para cada foto, procura o MENOR 'quality' do JPEG cujo resultado ainda tenha
SSIM >= alvo em relacao ao original. Preserva EXIF, perfil ICC e orientacao.

Uso GUI:  python jpegslim.py
Uso CLI:  python jpegslim.py -i fotos/ -o saida/ --level high
          python jpegslim.py -i foto.jpg --level ultra --overwrite

Dependencias: pip install pillow numpy
"""
import argparse
import io
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

# Nivel -> SSIM alvo (luminancia). Mais alto = mais fiel, arquivo maior.
LEVELS = {
    "medium": 0.970,
    "high": 0.985,
    "ultra": 0.993,
}
JPEG_EXT = {".jpg", ".jpeg", ".jpe"}
MIN_Q, MAX_Q = 45, 95


# ----------------------------------------------------------------- SSIM ----
def _box(x, w):
    c = np.cumsum(np.cumsum(x, axis=0, dtype=np.float64), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    s = c[w:, w:] - c[:-w, w:] - c[w:, :-w] + c[:-w, :-w]
    return s / (w * w)


def ssim(a: Image.Image, b: Image.Image, w: int = 7, strip: int = 512) -> float:
    """SSIM medio entre duas imagens (em escala de cinza), processado em faixas
    para nao estourar memoria em fotos grandes."""
    A = np.asarray(a.convert("L"), dtype=np.float32)
    B = np.asarray(b.convert("L"), dtype=np.float32)
    h, wd = A.shape
    if h < w or wd < w:
        return 1.0 if np.array_equal(A, B) else float(1 - np.mean(np.abs(A - B)) / 255)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    total, count = 0.0, 0
    r = 0
    while r < h - w + 1:
        r1 = min(r + strip, h - w + 1)
        x = A[r:r1 + w - 1].astype(np.float64)
        y = B[r:r1 + w - 1].astype(np.float64)
        mx, my = _box(x, w), _box(y, w)
        vx = _box(x * x, w) - mx * mx
        vy = _box(y * y, w) - my * my
        cxy = _box(x * y, w) - mx * my
        m = ((2 * mx * my + C1) * (2 * cxy + C2)) / (
            (mx * mx + my * my + C1) * (vx + vy + C2))
        total += float(m.sum())
        count += m.size
        r = r1
    return total / count


# ------------------------------------------------------------ otimizacao ----
def _encode(img, q, exif, icc, subsampling):
    buf = io.BytesIO()
    kw = dict(format="JPEG", quality=q, optimize=True, progressive=True)
    if exif:
        kw["exif"] = exif
    if icc:
        kw["icc_profile"] = icc
    if subsampling is not None:
        kw["subsampling"] = subsampling
    img.save(buf, **kw)
    return buf.getvalue()


def optimize_file(src, dst, target=0.985, keep_metadata=True):
    """Otimiza um arquivo. Retorna (tamanho_antes, tamanho_depois, qualidade, ssim)."""
    src, dst = Path(src), Path(dst)
    before = src.stat().st_size
    with Image.open(src) as im:
        im.load()
        exif = im.info.get("exif") if keep_metadata else None
        icc = im.info.get("icc_profile") if keep_metadata else None
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        orig = im.copy()

    # Busca binaria: menor q com ssim >= alvo
    lo, hi = MIN_Q, MAX_Q
    best = None  # (q, data, s)
    while lo <= hi:
        mid = (lo + hi) // 2
        data = _encode(orig, mid, exif, icc, None)
        dec = Image.open(io.BytesIO(data))
        s = ssim(orig, dec)
        if s >= target:
            best = (mid, data, s)
            hi = mid - 1
        else:
            lo = mid + 1
    if best is None:  # nem MAX_Q atinge o alvo: usa o maximo
        data = _encode(orig, MAX_Q, exif, icc, None)
        best = (MAX_Q, data, ssim(orig, Image.open(io.BytesIO(data))))

    q, data, s = best
    dst.parent.mkdir(parents=True, exist_ok=True)
    if len(data) >= before:
        # Ja estava bem comprimido: nao piora, copia o original.
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        return before, before, 0, 1.0
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dst)
    return before, len(data), q, s


def collect(paths, recursive=True):
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            out += [f for f in it if f.suffix.lower() in JPEG_EXT]
        elif p.suffix.lower() in JPEG_EXT and p.exists():
            out.append(p)
    return sorted(set(out))


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


# ------------------------------------------------------------------ CLI ----
def run_cli(args):
    files = collect(args.input)
    if not files:
        print("Nenhum JPEG encontrado.")
        return 1
    target = LEVELS[args.level]
    tb = ta = 0
    base = Path(args.input[0]) if Path(args.input[0]).is_dir() else None
    for f in files:
        if args.overwrite:
            dst = f
        else:
            outdir = Path(args.output) if args.output else f.parent
            rel = f.relative_to(base) if base else Path(f.name)
            dst = outdir / rel
            if not args.output:
                dst = dst.with_name(f.stem + "_opt" + f.suffix)
        b, a, q, s = optimize_file(f, dst, target)
        tb, ta = tb + b, ta + a
        print(f"{f.name}: {human(b)} -> {human(a)} "
              f"(-{100 * (1 - a / b):.0f}%) q={q} ssim={s:.4f}")
    print(f"\nTotal: {human(tb)} -> {human(ta)} (-{100 * (1 - ta / tb):.0f}%)")
    return 0


# ------------------------------------------------------------------ GUI ----
def run_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("JPEGSlim")
    root.geometry("820x520")

    files, state = [], {"out": "", "running": False}
    level = tk.StringVar(value="high")
    mode = tk.StringVar(value="suffix")
    keep = tk.BooleanVar(value=True)

    top = ttk.Frame(root, padding=8)
    top.pack(fill="x")
    cols = ("nome", "antes", "depois", "economia", "q", "status")
    tree = ttk.Treeview(root, columns=cols, show="headings", height=12)
    for c, w in zip(cols, (300, 90, 90, 90, 50, 150)):
        tree.heading(c, text=c.capitalize())
        tree.column(c, width=w, anchor="w" if c == "nome" else "center")
    tree.pack(fill="both", expand=True, padx=8)

    def add(paths):
        new = [f for f in collect(paths) if f not in files]
        for f in new:
            files.append(f)
            tree.insert("", "end", iid=str(f),
                        values=(f.name, human(f.stat().st_size), "", "", "", "Na fila"))

    def add_files():
        add(filedialog.askopenfilenames(
            filetypes=[("JPEG", "*.jpg *.jpeg *.jpe")]))

    def add_folder():
        d = filedialog.askdirectory()
        if d:
            add([d])

    def clear():
        if not state["running"]:
            files.clear()
            tree.delete(*tree.get_children())

    def pick_out():
        d = filedialog.askdirectory()
        if d:
            state["out"] = d
            mode.set("folder")
            out_lbl.config(text=d)

    ttk.Button(top, text="Adicionar fotos", command=add_files).pack(side="left")
    ttk.Button(top, text="Adicionar pasta", command=add_folder).pack(side="left", padx=4)
    ttk.Button(top, text="Limpar", command=clear).pack(side="left")

    opt = ttk.LabelFrame(root, text="Opcoes", padding=8)
    opt.pack(fill="x", padx=8, pady=6)
    ttk.Label(opt, text="Qualidade:").grid(row=0, column=0, sticky="w")
    for i, (k, t) in enumerate((("medium", "Media (menor arquivo)"),
                                ("high", "Alta"), ("ultra", "Ultra (mais fiel)"))):
        ttk.Radiobutton(opt, text=t, value=k, variable=level).grid(row=0, column=i + 1, padx=6)
    ttk.Label(opt, text="Saida:").grid(row=1, column=0, sticky="w")
    ttk.Radiobutton(opt, text="Sufixo _opt ao lado", value="suffix", variable=mode).grid(row=1, column=1)
    ttk.Radiobutton(opt, text="Substituir originais", value="over", variable=mode).grid(row=1, column=2)
    ttk.Radiobutton(opt, text="Outra pasta:", value="folder", variable=mode).grid(row=1, column=3)
    ttk.Button(opt, text="Escolher...", command=pick_out).grid(row=1, column=4)
    out_lbl = ttk.Label(opt, text="")
    out_lbl.grid(row=2, column=1, columnspan=4, sticky="w")
    ttk.Checkbutton(opt, text="Manter EXIF/ICC", variable=keep).grid(row=0, column=4)

    bar = ttk.Progressbar(root)
    bar.pack(fill="x", padx=8)
    total_lbl = ttk.Label(root, text="")
    total_lbl.pack(anchor="w", padx=8)

    def start():
        if state["running"] or not files:
            return
        if mode.get() == "over" and not messagebox.askyesno(
                "Confirmar", "Os originais serao substituidos. Continuar?"):
            return
        if mode.get() == "folder" and not state["out"]:
            messagebox.showwarning("Saida", "Escolha a pasta de saida.")
            return
        state["running"] = True
        target = LEVELS[level.get()]
        bar["maximum"], bar["value"] = len(files), 0
        acc = {"b": 0, "a": 0}
        lock = threading.Lock()

        def dest(f):
            if mode.get() == "over":
                return f
            if mode.get() == "folder":
                return Path(state["out"]) / f.name
            return f.with_name(f.stem + "_opt" + f.suffix)

        def work(f):
            iid = str(f)
            try:
                root.after(0, lambda: tree.set(iid, "status", "Otimizando..."))
                b, a, q, s = optimize_file(f, dest(f), target, keep.get())
                with lock:
                    acc["b"] += b
                    acc["a"] += a
                st = "Ja otimizado" if q == 0 else f"OK (SSIM {s:.3f})"
                root.after(0, lambda: tree.item(iid, values=(
                    f.name, human(b), human(a), f"-{100 * (1 - a / b):.0f}%", q or "-", st)))
            except Exception as e:  # noqa
                root.after(0, lambda: tree.set(iid, "status", f"Erro: {e}"))
            root.after(0, lambda: bar.step(1))

        def runner():
            with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
                list(ex.map(work, list(files)))
            def done():
                state["running"] = False
                if acc["b"]:
                    total_lbl.config(
                        text=f"Total: {human(acc['b'])} -> {human(acc['a'])} "
                             f"(-{100 * (1 - acc['a'] / acc['b']):.0f}%)")
            root.after(0, done)

        threading.Thread(target=runner, daemon=True).start()

    ttk.Button(root, text="Otimizar", command=start).pack(pady=6)
    root.mainloop()


def main():
    p = argparse.ArgumentParser(description="JPEGSlim - otimizador de JPEG")
    p.add_argument("-i", "--input", nargs="+", help="arquivos ou pastas")
    p.add_argument("-o", "--output", help="pasta de saida")
    p.add_argument("--level", choices=LEVELS, default="high")
    p.add_argument("--overwrite", action="store_true", help="substitui os originais")
    args = p.parse_args()
    if args.input:
        sys.exit(run_cli(args))
    run_gui()


if __name__ == "__main__":
    main()
