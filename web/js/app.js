// Client web : appelle l'API (/search et /search/upload) et affiche la grille de résultats.
const $ = (id) => document.getElementById(id);
const MAX_BYTES = 10 * 1024 * 1024;
let queryImage = null; // id d'une image de la base (clic sur un résultat)
let uploadFile = null; // fichier envoyé depuis l'ordinateur
let previewUrl = null;

function setStatus(message, isError = false) {
    $("status").textContent = message;
    $("status").classList.toggle("error", isError);
}

function applyMode() {
    const mode = $("mode").value;
    $("textbox").hidden = mode === "image";
    $("imagebox").hidden = mode === "text";
    $("descs").hidden = mode === "text";
    $("opwrap").hidden = mode !== "both";
}

function showQuery(src, label) {
    $("qimg").src = src;
    $("qname").textContent = label;
    $("query").hidden = false;
}

function clearQuery() {
    queryImage = null;
    uploadFile = null;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    $("file").value = "";
    $("query").hidden = true;
}

function handleFile(file) {
    if (!file) return;
    if (!["image/png", "image/jpeg"].includes(file.type)) {
        setStatus("Format non accepté : utilisez une image PNG, JPG ou JPEG.", true);
        return;
    }
    if (file.size > MAX_BYTES) {
        setStatus("Image trop lourde (10 Mo maximum).", true);
        return;
    }
    clearQuery();
    uploadFile = file;
    previewUrl = URL.createObjectURL(file);
    showQuery(previewUrl, file.name);
    setStatus("");
}

function render(results) {
    const grid = $("results");
    grid.replaceChildren();
    for (const hit of results) {
        const card = document.createElement("figure");
        card.className = "item";

        const btn = document.createElement("button");
        btn.type = "button";
        btn.title = "Rechercher les images similaires";
        const img = document.createElement("img");
        img.src = hit.url;
        img.alt = hit.id;
        img.loading = "lazy";
        btn.appendChild(img);
        btn.addEventListener("click", () => {
            clearQuery();
            queryImage = hit.id;
            showQuery(hit.url, hit.id);
            if ($("mode").value === "text") {
                $("mode").value = "image";
                applyMode();
            }
            runSearch();
            window.scrollTo({ top: 0, behavior: "smooth" });
        });

        const info = document.createElement("div");
        const cat = document.createElement("strong");
        cat.textContent = hit.category || hit.id;
        const detail = document.createElement("span");
        if (hit.distance !== null && hit.distance !== undefined) {
            detail.textContent = "distance " + hit.distance.toFixed(4);
        } else if (hit.score !== null && hit.score !== undefined) {
            detail.textContent = "score " + hit.score.toFixed(2);
        }
        info.append(cat, detail);
        card.append(btn, info);
        grid.appendChild(card);
    }
}

async function runSearch() {
    const mode = $("mode").value;
    const needText = mode !== "image";
    const needImage = mode !== "text";
    const text = $("text").value.trim();
    const descriptors = [...document.querySelectorAll("#descs input:checked")].map((c) => c.value);
    const base = { dataset: $("dataset").value, operator: $("operator").value, k: Number($("k").value) || 12 };

    if (needText && !text) return setStatus("Saisissez des mots-clés.", true);
    if (needImage && !queryImage && !uploadFile) {
        return setStatus("Choisissez une image : envoi depuis l'ordinateur ou clic sur un résultat.", true);
    }
    if (needImage && descriptors.length === 0) return setStatus("Cochez au moins un descripteur.", true);

    setStatus("Recherche en cours…");
    try {
        let res;
        if (needImage && uploadFile) {
            const fd = new FormData();
            fd.append("file", uploadFile);
            fd.append("dataset", base.dataset);
            fd.append("operator", base.operator);
            fd.append("k", String(base.k));
            fd.append("descriptors", descriptors.join(","));
            fd.append("text", needText ? text : "");
            res = await fetch("/search/upload", { method: "POST", body: fd });
        } else {
            res = await fetch("/search", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    ...base,
                    text: needText ? text : null,
                    query_image_id: needImage ? queryImage : null,
                    descriptors: needImage ? descriptors : [],
                }),
            });
        }
        const data = await res.json();
        if (!res.ok) {
            render([]);
            return setStatus(typeof data.detail === "string" ? data.detail : "Requête invalide.", true);
        }
        render(data.results);
        setStatus(data.count + " résultat(s)");
    } catch (err) {
        render([]);
        setStatus("Impossible de joindre l'API. Est-elle démarrée ?", true);
    }
}

$("mode").addEventListener("change", applyMode);
$("k").addEventListener("input", () => ($("kout").textContent = $("k").value));
$("form").addEventListener("submit", (e) => {
    e.preventDefault();
    runSearch();
});
$("clear").addEventListener("click", clearQuery);
$("file").addEventListener("change", (e) => handleFile(e.target.files[0]));
const drop = $("drop");
drop.addEventListener("dragover", (e) => {
    e.preventDefault();
    drop.classList.add("over");
});
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    handleFile(e.dataTransfer.files[0]);
});
applyMode();