
from flask import Flask, request, send_file, render_template
from flask_cors import CORS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import io

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
  return render_template("index.html")
plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 14,
    "legend.fontsize": 11, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "black", "axes.linewidth": 1.0,
    "grid.color": "0.85", "grid.linewidth": 0.6,
})

LABEL_MAP = {
    "r0": r"$r_0$", "e2": r"$e^2$", "Zm": r"$Z_m$", "A1": r"$A_1$", "B1": r"$B_1$",
    "A2": r"$A_2$", "B2": r"$B_2$",
    "f1": r"$f_0'$", "f2": r"$f_0''$", "f3": r"$f_0'''$",
    "xi2": r"$\xi'^2$"
}


def compute(vals):
    r0, e2, Zm, A1, B1, A2, B2, f1, f2, f3, xi2 = (
        vals["r0"], vals["e2"], vals["Zm"], vals["A1"], vals["B1"],
        vals["A2"], vals["B2"], vals["f1"], vals["f2"], vals["f3"], vals["xi2"]
    )
    Zm2 = Zm**2
    P = e2 / (4 * r0**4)
    C1 = A1**2 / B1 if B1 != 0 else np.nan
    D1 = A1**3 / B1**2 if B1 != 0 else np.nan
    C2 = A2**2 / B2 if B2 != 0 else np.nan
    D2 = A2**3 / B2**2 if B2 != 0 else np.nan
    R2p = (D2 - 6*C2 + 15*A2 - 15*B2) / 16

    C11 = P*(-5.112*Zm2 + A1 + 0.5*A2 + 0.5*B2 + 9.3204*xi2)
    C12 = P*(0.226*Zm2 - B1 + 0.25*A2 - 1.25*B2 + 9.3204*xi2)
    C44 = P*(2.556*Zm2 + B1 + 0.25*A2 + 0.75*B2)
    K = (C11 + 2*C12) / 3

    C111 = P*(37.556*Zm2 + D1 - 3*C1 + 0.25*(C2-3*A2-9*B2) + 13.98*r0*r0*f2 - 89.303*r0*f1)
    C112 = P*(-4.836*Zm2 + 0.125*(C2-3*A2-3*B2) + 4.660*r0*r0*f2 - 18.640*r0*f1)
    C166 = P*(-7.166*Zm2 - 2*(B1+B2) + 0.125*(C2-3*A2+3*B2) + 5.564*r0*f1)
    C123 = P*(2.717*Zm2 + 16.692*r0*f1)
    C144 = P*(2.717*Zm2 + 5.564*r0*f1)
    C456 = P*(2.717*Zm2)

    C1111 = P*(-305.321*Zm2 + D1 - 6*C1 + 15*A1 + 2*R2p - 11.25*B2 + 18.6407*r0**3*f3 - 206.574*r0*r0*f2 + 863.498*r0*f1)
    C1112 = P*(17.9034*Zm2 + R2p + 4.6602*r0**3*f3 + 2.7116*r0*r0*f2 + 103.489*r0*f1)
    C1166 = P*(27.2234*Zm2 + 8*(B1+B2) + R2p + 5.564*r0*r0*f2 - 44.2513*r0*f1)
    C1122 = P*(22.4611*Zm2 + R2p - 50.2166*r0*r0*f2 - 27.158*r0*f1)
    C1266 = P*(27.1211*Zm2 + 4*(B1+B2) + R2p - 38.6873*r0*f1)
    C4444 = P*(32.9562*Zm2 + 9*(B1+B2) + R2p + 10.2242*r0*r0*f2 - 10.2242*r0*f1)
    C1123 = P*(-6.3406*Zm2 + 5.564*r0*r0*f2 - 22.5157*r0*f1)
    C1144 = P*(-6.3406*Zm2 + 5.564*r0*r0*f2 + 16.692*r0*f1)
    C1244 = P*(-6.3406*Zm2 - 8.4746*r0*f1)
    C1456 = P*(-6.3406*Zm2 + 10.8678*r0*f1)
    C4466 = P*(-4.0106*Zm2 + 2*(B1+B2) + 10.2242*r0*r0*f2 - 10.2229*r0*f1)

    Omega = -2.33*Zm2 + A1 + A2 + 27.961*r0*f1
    dK = -(13.975*Zm2 + C1 - 3*A1 + C2 - 3*A2 - 167.764*r0*f1 + 41.94*r0*r0*f2) / (3*Omega)
    dS = -(23.676*Zm2 + C1 + (C2+6*A2-6*B2)/4 - 51.07584*r0*f1 + 13.98*r0*r0*f2) / (2*Omega)
    dC44 = -(-11.389*Zm2 + A1 - 3*B1 + (C2+2*A2-10*B2)/4 + 44.6524*r0*r0*f1) / Omega
    dC111 = -(-3*C11 - 6*C12 + 3*C111 + C1111 + 2*C1112) / (3*K)
    dC112 = -(C11 + 2*C12 + 3*C112 + C1112 + C1122 + C1123) / (3*K)
    dC166 = -(-C11 - 2*C12 + 3*C166 + C1166 + 2*C1244) / (3*K)
    dC123 = -(-C11 - 2*C12 + 3*C123 + 3*C1123) / (3*K)
    dC144 = -(C11 + 2*C12 + 3*C144 + C1144 + 2*C1244) / (3*K)
    dC456 = -(-C11 - 2*C12 + 3*C456 + 3*C1456) / (3*K)

    return {
        "SOEC": {"C11": C11, "C12": C12, "C44": C44},
        "TOEC": {"C111": C111, "C112": C112, "C166": C166, "C123": C123, "C144": C144, "C456": C456},
        "FOEC": {"C1111": C1111, "C1112": C1112, "C1166": C1166, "C1122": C1122, "C1266": C1266,
                 "C4444": C4444, "C1123": C1123, "C1144": C1144, "C1244": C1244, "C1456": C1456, "C4466": C4466},
        "PRESSURE": {"dK_dP": dK, "dS_dP": dS, "dC44_dP": dC44, "dC111_dP": dC111,
                     "dC112_dP": dC112, "dC166_dP": dC166, "dC123_dP": dC123, "dC144_dP": dC144, "dC456_dP": dC456}
    }


def make_chart(sweep_param, sweep_min, sweep_max, family, base_inputs, n=80):
    xs = np.linspace(sweep_min, sweep_max, n)
    series = {}
    for x in xs:
        vals = dict(base_inputs)
        vals[sweep_param] = x
        out = compute(vals)[family]
        for k, v in out.items():
            series.setdefault(k, []).append(v)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    styles = ["-", "--", "-.", ":"]
    for i, (name, ys) in enumerate(series.items()):
        ax.plot(xs, ys, label="$" + name + "$", lw=2.0, linestyle=styles[i % len(styles)])

    ax.set_xlabel(LABEL_MAP.get(sweep_param, sweep_param))
    ax.set_ylabel("Elastic constant (normalized units)")
    ax.set_title(family + " vs " + LABEL_MAP.get(sweep_param, sweep_param))
    ax.grid(True, alpha=0.5)
    ax.legend(frameon=True, ncol=2 if len(series) > 4 else 1)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


@app.route("/chart", methods=["POST"])
def chart():
    payload = request.get_json()
    buf = make_chart(
        payload["sweep_param"], payload["sweep_min"], payload["sweep_max"],
        payload["family"], payload["inputs"]
    )
    fname = payload["family"].lower() + "_vs_" + payload["sweep_param"] + ".png"
    return send_file(buf, mimetype="image/png", as_attachment=True, download_name=fname)


if __name__ == "__main__":
    app.run(port=5001, debug=True)
