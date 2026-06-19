import ast, io, tokenize, sys, glob

def strip_py(src):
    lines = src.splitlines(keepends=True)
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type == tokenize.COMMENT:
            r, c0 = t.start; _, c1 = t.end
            lines[r-1] = lines[r-1][:c0] + lines[r-1][c1:]
    text = "".join(lines)
    tree = ast.parse(text)
    drop = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = node.body
            if (len(b) > 1 and isinstance(b[0], ast.Expr)
                    and isinstance(getattr(b[0], "value", None), ast.Constant)
                    and isinstance(b[0].value.value, str)):
                for ln in range(b[0].lineno, b[0].end_lineno + 1):
                    drop.add(ln)
    kept = [ln for i, ln in enumerate(text.splitlines(), 1) if i not in drop]
    return _collapse(kept)

def strip_tex(src):
    out = []
    for line in src.splitlines():
        res, i, n, comment = [], 0, len(line), False
        while i < n:
            if line[i] == "%" and (i == 0 or line[i-1] != "\\"):
                comment = True; break
            res.append(line[i]); i += 1
        code = "".join(res) if comment else line
        if comment and code.strip() == "":
            continue
        out.append(code.rstrip())
    return _collapse(out)

def _collapse(lines):
    out, prev_blank = [], False
    for ln in lines:
        ln = ln.rstrip()
        if ln == "":
            if not prev_blank: out.append("")
            prev_blank = True
        else:
            out.append(ln); prev_blank = False
    return "\n".join(out).strip() + "\n"

if __name__ == "__main__":
    pyf = glob.glob("src/**/*.py", recursive=True)
    for f in pyf:
        s = open(f).read()
        open(f, "w").write(strip_py(s))
        print("py  ", f)
    for f in ["paper/main.tex", "paper/results_macros.tex"]:
        s = open(f).read()
        open(f, "w").write(strip_tex(s))
        print("tex ", f)
