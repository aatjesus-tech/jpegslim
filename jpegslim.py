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
import json
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


# --------------------------------------------------------- estatisticas ----
def stats_file():
    base = os.environ.get("APPDATA") or str(Path.home() / ".config")
    return Path(base) / "JPEGSlim" / "stats.json"


def load_stats():
    try:
        d = json.loads(stats_file().read_text(encoding="utf-8"))
        return {"saved": int(d.get("saved", 0)), "photos": int(d.get("photos", 0))}
    except Exception:
        return {"saved": 0, "photos": 0}


def save_stats(st):
    try:
        p = stats_file()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(st), encoding="utf-8")
        os.replace(tmp, p)
    except Exception:
        pass  # nunca deixa uma falha de gravacao atrapalhar a otimizacao


def human_total(n):
    """Formata o acumulado: KB, MB e depois GB com 2 casas."""
    if n < 1024 ** 2:
        return f"{n / 1024:.0f} KB"
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f} MB"
    return f"{n / 1024 ** 3:.2f} GB"


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
ACCENT = "#7c5cff"
ACCENT_HOVER = "#6a49f0"
BG = "#0e1016"
CARD = "#171a23"
CARD_2 = "#1e2230"
TEXT_DIM = "#8b93a7"
GREEN = "#34d399"
RED = "#f87171"


def build_app():
    import queue
    from tkinter import filedialog, messagebox, ttk

    import customtkinter as ctk

    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
        have_dnd = True
    except Exception:  # sem arrastar-e-soltar, o app continua funcionando
        have_dnd = False

    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    if have_dnd:
        class Root(ctk.CTk, TkinterDnD.DnDWrapper):
            def __init__(self):
                super().__init__()
                self.TkdndVersion = TkinterDnD._require(self)
    else:
        Root = ctk.CTk

    class App(Root):
        LEVEL_LABELS = {"Média": "medium", "Alta": "high", "Ultra": "ultra"}
        OUT_SUFFIX = "Salvar ao lado (_opt)"
        OUT_OVER = "Substituir originais"
        OUT_FOLDER = "Escolher pasta..."

        def __init__(self):
            super().__init__()
            self.title("JPEGSlim")
            self.geometry("900x720")
            self.minsize(760, 560)
            self.configure(fg_color=BG)

            self.files = []          # Path
            self.done = set()        # Path ja processados
            self.out_dir = None
            self.running = False
            self.q = queue.Queue()
            self.stats = load_stats()
            self.acc = {"b": 0, "a": 0}
            self.total_todo = 0
            self.total_done = 0

            self._build_ui()
            self.after(100, self._poll)

        # ---------------------------------------------------------- UI ----
        def _build_ui(self):
            head = ctk.CTkFrame(self, fg_color="transparent")
            head.pack(fill="x", padx=28, pady=(22, 8))
            ctk.CTkLabel(head, text="JPEGSlim",
                         font=ctk.CTkFont(size=28, weight="bold")).pack(side="left")
            ctk.CTkLabel(head, text="  Fotos mais leves, sem perda visível",
                         font=ctk.CTkFont(size=14), text_color=TEXT_DIM).pack(
                side="left", pady=(10, 0))

            # total acumulado (persistente entre execucoes)
            tot = ctk.CTkFrame(head, fg_color=CARD, corner_radius=12)
            tot.pack(side="right")
            ctk.CTkLabel(tot, text="TOTAL ECONOMIZADO", text_color=TEXT_DIM,
                         font=ctk.CTkFont(size=10, weight="bold")).pack(
                padx=16, pady=(8, 0))
            self.total_lbl = ctk.CTkLabel(tot, text="0 KB", text_color=GREEN,
                                          font=ctk.CTkFont(size=22, weight="bold"))
            self.total_lbl.pack(padx=16)
            self.total_sub = ctk.CTkLabel(tot, text="", text_color=TEXT_DIM,
                                          font=ctk.CTkFont(size=11))
            self.total_sub.pack(padx=16, pady=(0, 8))
            self._refresh_total()

            # zona de soltar
            self.zone = ctk.CTkFrame(self, fg_color=CARD, corner_radius=18,
                                     border_width=2, border_color=ACCENT, height=130)
            self.zone.pack(fill="x", padx=28, pady=8)
            self.zone.pack_propagate(False)
            self.zone_title = ctk.CTkLabel(
                self.zone, text="Arraste suas fotos ou pastas aqui",
                font=ctk.CTkFont(size=18, weight="bold"))
            self.zone_title.pack(pady=(26, 2))
            self.zone_sub = ctk.CTkLabel(
                self.zone,
                text="ou clique para escolher  •  JPG / JPEG" if have_dnd else
                     "Clique para escolher as fotos  •  JPG / JPEG",
                text_color=TEXT_DIM)
            self.zone_sub.pack()
            for w in (self.zone, self.zone_title, self.zone_sub):
                w.bind("<Button-1>", lambda e: self.pick_files())
                w.configure(cursor="hand2") if w is self.zone else None

            # barra de acoes da lista
            bar = ctk.CTkFrame(self, fg_color="transparent")
            bar.pack(fill="x", padx=28, pady=(6, 0))
            self.count_lbl = ctk.CTkLabel(bar, text="Nenhuma foto na fila",
                                          text_color=TEXT_DIM)
            self.count_lbl.pack(side="left")
            ctk.CTkButton(bar, text="Limpar", width=80, height=28, fg_color=CARD_2,
                          hover_color="#2a3045", command=self.clear).pack(side="right")
            ctk.CTkButton(bar, text="+ Pasta", width=80, height=28, fg_color=CARD_2,
                          hover_color="#2a3045", command=self.pick_folder).pack(
                side="right", padx=6)

            # lista
            box = ctk.CTkFrame(self, fg_color=CARD, corner_radius=14)
            box.pack(fill="both", expand=True, padx=28, pady=8)
            style = ttk.Style()
            style.theme_use("clam")
            style.configure("J.Treeview", background=CARD, fieldbackground=CARD,
                            foreground="#e6e9f2", rowheight=30, borderwidth=0,
                            font=("Segoe UI", 10))
            style.configure("J.Treeview.Heading", background=CARD_2,
                            foreground=TEXT_DIM, borderwidth=0, relief="flat",
                            font=("Segoe UI", 9, "bold"))
            style.map("J.Treeview", background=[("selected", "#2a2f45")])
            style.configure("J.Treeview", bordercolor=CARD, lightcolor=CARD,
                            darkcolor=CARD)
            style.layout("J.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])
            style.map("J.Treeview.Heading", background=[("active", CARD_2)])
            style.configure("Vertical.TScrollbar", background="#2a3045",
                            troughcolor=CARD, borderwidth=0, arrowcolor=TEXT_DIM,
                            bordercolor=CARD, lightcolor="#2a3045", darkcolor="#2a3045")
            style.map("Vertical.TScrollbar", background=[("active", "#3a4260")])
            cols = ("nome", "antes", "depois", "eco", "status")
            self.tree = ttk.Treeview(box, columns=cols, show="headings",
                                     style="J.Treeview", selectmode="extended", height=4)
            heads = (("nome", "Arquivo", 330, "w"), ("antes", "Antes", 90, "center"),
                     ("depois", "Depois", 90, "center"),
                     ("eco", "Economia", 90, "center"), ("status", "Status", 170, "center"))
            for c, t, w, a in heads:
                self.tree.heading(c, text=t)
                self.tree.column(c, width=w, anchor=a, stretch=(c == "nome"))
            sb = ttk.Scrollbar(box, orient="vertical", command=self.tree.yview)
            self.tree.configure(yscrollcommand=sb.set)
            sb.pack(side="right", fill="y", padx=(0, 4), pady=6)
            self.tree.pack(fill="both", expand=True, padx=(8, 0), pady=6)
            self.tree.tag_configure("ok", foreground=GREEN)
            self.tree.tag_configure("err", foreground=RED)
            self.tree.bind("<Delete>", lambda e: self.remove_selected())

            # opcoes
            opt = ctk.CTkFrame(self, fg_color=CARD, corner_radius=14)
            opt.pack(side="bottom", before=box, fill="x", padx=28, pady=(0, 8))
            self._opt = opt
            ctk.CTkLabel(opt, text="Qualidade", text_color=TEXT_DIM).grid(
                row=0, column=0, padx=(16, 8), pady=(12, 0), sticky="w")
            ctk.CTkLabel(opt, text="Saída", text_color=TEXT_DIM).grid(
                row=0, column=1, padx=8, pady=(12, 0), sticky="w")
            self.level = ctk.CTkSegmentedButton(
                opt, values=list(self.LEVEL_LABELS), selected_color=ACCENT,
                selected_hover_color=ACCENT_HOVER, fg_color=CARD_2)
            self.level.set("Alta")
            self.level.grid(row=1, column=0, padx=(16, 8), pady=(4, 14), sticky="w")
            self.out_menu = ctk.CTkOptionMenu(
                opt, values=[self.OUT_SUFFIX, self.OUT_OVER, self.OUT_FOLDER],
                command=self._out_changed, fg_color=CARD_2, button_color=CARD_2,
                button_hover_color="#2a3045", width=210)
            self.out_menu.set(self.OUT_SUFFIX)
            self.out_menu.grid(row=1, column=1, padx=8, pady=(4, 14), sticky="w")
            self.keep = ctk.CTkSwitch(opt, text="Manter EXIF / cores", progress_color=ACCENT)
            self.keep.select()
            self.keep.grid(row=1, column=2, padx=16, pady=(4, 14), sticky="e")
            opt.grid_columnconfigure(2, weight=1)

            # rodape
            foot = ctk.CTkFrame(self, fg_color="transparent")
            foot.pack(side="bottom", before=self._opt, fill="x", padx=28, pady=(0, 18))
            left = ctk.CTkFrame(foot, fg_color="transparent")
            left.pack(side="left", fill="x", expand=True)
            self.progress = ctk.CTkProgressBar(left, progress_color=ACCENT, height=10)
            self.progress.set(0)
            self.progress.pack(fill="x", padx=(0, 18), pady=(6, 6))
            self.summary = ctk.CTkLabel(left, text="", anchor="w", text_color=TEXT_DIM)
            self.summary.pack(fill="x")
            self.btn = ctk.CTkButton(foot, text="Otimizar", width=170, height=46,
                                     corner_radius=12, fg_color=ACCENT,
                                     hover_color=ACCENT_HOVER,
                                     font=ctk.CTkFont(size=16, weight="bold"),
                                     command=self.start)
            self.btn.pack(side="right")

            if have_dnd:
                for w in (self, self.zone, self.zone_title, self.zone_sub, self.tree):
                    w.drop_target_register(DND_FILES)
                    w.dnd_bind("<<Drop>>", self._on_drop)
                    w.dnd_bind("<<DropEnter>>", lambda e: self.zone.configure(fg_color=CARD_2))
                    w.dnd_bind("<<DropLeave>>", lambda e: self.zone.configure(fg_color=CARD))

        def _refresh_total(self):
            self.total_lbl.configure(text=human_total(self.stats["saved"]))
            n = self.stats["photos"]
            self.total_sub.configure(text=f"em {n} foto(s) otimizada(s)" if n else
                                     "desde a primeira vez que usou")

        # ------------------------------------------------------- fila ----
        def _on_drop(self, event):
            self.zone.configure(fg_color=CARD)
            self.add_paths(self.tk.splitlist(event.data))
            return event.action

        def add_paths(self, paths):
            if self.running:
                return
            n = 0
            for f in collect(paths):
                if f in self.files:
                    continue
                self.files.append(f)
                try:
                    size = human(f.stat().st_size)
                except OSError:
                    size = "?"
                self.tree.insert("", "end", iid=str(f),
                                 values=(f.name, size, "", "", "Na fila"))
                n += 1
            self._update_count()
            return n

        def _update_count(self):
            pend = len([f for f in self.files if f not in self.done])
            self.count_lbl.configure(
                text=f"{pend} foto(s) na fila" if pend else "Nenhuma foto na fila")

        def pick_files(self):
            if self.running:
                return
            fs = filedialog.askopenfilenames(
                title="Escolha as fotos",
                filetypes=[("JPEG", "*.jpg *.jpeg *.jpe *.JPG *.JPEG")])
            if fs:
                self.add_paths(fs)

        def pick_folder(self):
            if self.running:
                return
            d = filedialog.askdirectory(title="Escolha uma pasta")
            if d:
                self.add_paths([d])

        def remove_selected(self):
            if self.running:
                return
            for iid in self.tree.selection():
                p = Path(iid)
                if p in self.files:
                    self.files.remove(p)
                self.done.discard(p)
                self.tree.delete(iid)
            self._update_count()

        def clear(self):
            if self.running:
                return
            self.files.clear()
            self.done.clear()
            self.tree.delete(*self.tree.get_children())
            self.progress.set(0)
            self.summary.configure(text="")
            self._update_count()

        def _out_changed(self, choice):
            if choice == self.OUT_FOLDER:
                d = filedialog.askdirectory(title="Pasta de saída")
                if d:
                    self.out_dir = Path(d)
                    self.out_menu.set(f"Pasta: …{str(d)[-22:]}")
                else:
                    self.out_menu.set(self.OUT_SUFFIX)

        def _dest(self, f):
            cur = self.out_menu.get()
            if cur == self.OUT_OVER:
                return f
            if self.out_dir and cur.startswith("Pasta:"):
                return self.out_dir / f.name
            return f.with_name(f.stem + "_opt" + f.suffix)

        # --------------------------------------------------- processar ----
        def start(self):
            if self.running:
                return
            todo = [f for f in self.files if f not in self.done]
            if not todo:
                messagebox.showinfo("JPEGSlim", "Adicione fotos antes de otimizar.")
                return
            if self.out_menu.get() == self.OUT_OVER and not messagebox.askyesno(
                    "Substituir originais",
                    "Os arquivos originais serão substituídos pelas versões otimizadas.\n"
                    "Deseja continuar?"):
                return
            self.running = True
            self.btn.configure(state="disabled", text="Otimizando...")
            target = LEVELS[self.LEVEL_LABELS[self.level.get()]]
            keep = bool(self.keep.get())
            self.acc = {"b": 0, "a": 0}
            self.total_todo, self.total_done = len(todo), 0
            self.progress.set(0)
            self.summary.configure(text=f"Processando 0 de {len(todo)}...")
            dests = {f: self._dest(f) for f in todo}

            def work(f):
                self.q.put(("start", f))
                try:
                    r = optimize_file(f, dests[f], target, keep)
                    self.q.put(("ok", f, r))
                except Exception as e:  # noqa
                    self.q.put(("err", f, str(e)))

            def runner():
                workers = max(1, (os.cpu_count() or 2) - 1)
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    list(ex.map(work, todo))
                self.q.put(("end",))

            threading.Thread(target=runner, daemon=True).start()

        def _poll(self):
            try:
                while True:
                    m = self.q.get_nowait()
                    kind = m[0]
                    if kind == "start":
                        self.tree.set(str(m[1]), "status", "Otimizando...")
                    elif kind == "ok":
                        f, (b, a, qv, _s) = m[1], m[2]
                        self.acc["b"] += b
                        self.acc["a"] += a
                        self.done.add(f)
                        if a < b:
                            self.stats["saved"] += b - a
                            self.stats["photos"] += 1
                            save_stats(self.stats)
                            self._refresh_total()
                        eco = f"-{100 * (1 - a / b):.0f}%" if a < b else "0%"
                        st = "Já otimizada" if qv == 0 else f"Concluída (q{qv})"
                        self.tree.item(str(f), values=(f.name, human(b), human(a), eco, st),
                                       tags=("ok",))
                        self._tick()
                    elif kind == "err":
                        self.tree.set(str(m[1]), "status", "Erro")
                        self.tree.item(str(m[1]), tags=("err",))
                        self._tick()
                    elif kind == "end":
                        self.running = False
                        self.btn.configure(state="normal", text="Otimizar")
                        b, a = self.acc["b"], self.acc["a"]
                        if b:
                            self.summary.configure(
                                text=f"Pronto!  {human(b)} → {human(a)}  "
                                     f"(economia de {human(b - a)}, -{100 * (1 - a / b):.0f}%)",
                                text_color=GREEN)
                        self._update_count()
            except queue.Empty:
                pass
            self.after(100, self._poll)

        def _tick(self):
            self.total_done += 1
            self.progress.set(self.total_done / max(1, self.total_todo))
            self.summary.configure(
                text=f"Processando {self.total_done} de {self.total_todo}...",
                text_color=TEXT_DIM)

    return App()


def run_gui():
    app = build_app()
    app.mainloop()


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
