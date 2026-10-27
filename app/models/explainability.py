import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

FEATURE_NAMES = ["NDVI", "NDRE", "GNDVI"]


def max_membership(distances, m=2.0):
    """Grado de pertenencia FCM de la zona asignada: 1 / sum_j (d_min / d_j)^(2/(m-1))."""
    d = np.fmax(distances, 1e-10)
    d_min = d.min(axis=1, keepdims=True)
    return (1.0 / ((d_min / d) ** (2.0 / (m - 1.0))).sum(axis=1)).astype(np.float32)


def local_attribution(z, centers, feature_names=FEATURE_NAMES):
    """
    Explicacion aditiva y exacta de por que un pixel (estandarizado) quedo en su zona y no en la
    segunda mas cercana: d_B^2 - d_A^2 = sum_f [(z_f - c_Bf)^2 - (z_f - c_Af)^2].
    """
    d2 = ((z[None, :] - centers) ** 2).sum(axis=1)
    a, b = np.argsort(d2)[:2]
    contrib = (z - centers[b]) ** 2 - (z - centers[a]) ** 2
    return {
        "zona_asignada": int(a),
        "segunda_zona": int(b),
        "ventaja_total": round(float(d2[b] - d2[a]), 4),
        "aportes": {f: round(float(v), 4) for f, v in zip(feature_names, contrib)},
    }


class ZoneExplainer:
    """
    Explica la zonificacion FCM con cinco piezas:
      1. prototipos: el perfil medio (centroide) de cada zona en unidades originales;
      2. importancia por permutacion: % de pixeles que cambian de zona al desordenar cada indice;
      3. modelo sustituto: arbol de decision poco profundo que imita al FCM, con su fidelidad y reglas;
      4. explicaciones locales exactas de pixeles tipicos y de transicion;
      5. incertidumbre: grado de pertenencia y proporcion de pixeles de transicion.
    """

    def __init__(self, sample_size=200_000, transition_threshold=0.6, tree_depth=3, random_state=42):
        self.sample_size = sample_size
        self.transition_threshold = transition_threshold
        self.tree_depth = tree_depth
        self.random_state = random_state

    def explain(self, features, labels, centers, scaler_mean, scaler_scale, fcm_m=2.0,
                memberships=None, feature_names=FEATURE_NAMES):
        rng = np.random.default_rng(self.random_state)
        centers = np.asarray(centers, dtype=np.float64)
        mean = np.asarray(scaler_mean, dtype=np.float64)
        scale = np.asarray(scaler_scale, dtype=np.float64)
        n = len(features)
        s = rng.choice(n, size=min(self.sample_size, n), replace=False)
        X = features[s].astype(np.float64)
        Z = (X - mean) / scale
        y = labels[s]
        d = np.linalg.norm(Z[:, None, :] - centers[None], axis=2)
        u_sample = max_membership(d, fcm_m)

        return {
            "prototipos": self._prototypes(centers, mean, scale, labels, feature_names),
            "importancia_permutacion": self._permutation(Z, y, centers, rng, feature_names),
            "modelo_sustituto": self._surrogate(X, y, rng, feature_names),
            "explicaciones_locales": self._local_examples(Z, X, y, u_sample, centers, feature_names),
            "incertidumbre": self._uncertainty(memberships if memberships is not None else u_sample,
                                               len(centers)),
            "nota": (
                "La explicacion local es aditiva y exacta para un modelo de prototipos (asignacion al "
                "centroide mas cercano): los aportes de los indices suman la ventaja de la zona asignada "
                "sobre la segunda, la misma propiedad de precision local que garantizan los valores SHAP."
            ),
        }

    @staticmethod
    def _prototypes(centers, mean, scale, labels, names):
        proto = centers * scale + mean
        counts = np.bincount(labels.astype(np.int64), minlength=len(centers))
        out = []
        for c in np.argsort(-proto[:, 0]):
            out.append({
                "cluster_id": int(c),
                **{n.lower(): round(float(v), 4) for n, v in zip(names, proto[c])},
                "pct_pixeles": round(float(counts[c] * 100.0 / max(counts.sum(), 1)), 2),
            })
        return out

    @staticmethod
    def _permutation(Z, y, centers, rng, names):
        out = {}
        for j, name in enumerate(names):
            Zp = Z.copy()
            Zp[:, j] = rng.permutation(Zp[:, j])
            yp = np.argmin(np.linalg.norm(Zp[:, None, :] - centers[None], axis=2), axis=1)
            out[name] = round(float(np.mean(yp != y) * 100.0), 2)
        return out

    def _surrogate(self, X, y, rng, names):
        idx = rng.permutation(len(X))
        half = len(X) // 2
        tr, te = idx[:half], idx[half:]
        tree = DecisionTreeClassifier(max_depth=self.tree_depth, random_state=self.random_state)
        tree.fit(X[tr], y[tr])
        return {
            "profundidad": self.tree_depth,
            "fidelidad_pct": round(float(tree.score(X[te], y[te]) * 100.0), 2),
            "importancia": {n: round(float(v), 4) for n, v in zip(names, tree.feature_importances_)},
            "reglas": export_text(tree, feature_names=list(names), decimals=3),
        }

    @staticmethod
    def _local_examples(Z, X, y, u, centers, names):
        ejemplos = []
        for c in range(len(centers)):
            m = np.nonzero(y == c)[0]
            if m.size:
                i = m[np.argmax(u[m])]
                ejemplos.append(("tipico", i))
        order = np.argsort(np.abs(u - 0.5))[:3]
        ejemplos += [("transicion", i) for i in order]
        out = []
        for tipo, i in ejemplos:
            out.append({
                "tipo": tipo,
                **{n.lower(): round(float(v), 4) for n, v in zip(names, X[i])},
                "pertenencia_max": round(float(u[i]), 3),
                **local_attribution(Z[i], centers, names),
            })
        return out

    def _uncertainty(self, u, k):
        u = np.asarray(u)
        return {
            "pertenencia_max_media": round(float(u.mean()), 4),
            "pct_pixeles_transicion": round(float(np.mean(u < self.transition_threshold) * 100.0), 2),
            "umbral_transicion": self.transition_threshold,
            "minimo_teorico": round(1.0 / k, 4),
        }
