// Client web : appelle l'API et affiche les resultats.
const $ = (id) => document.getElementById(id);
const MAX_BYTES = 10 * 1024 * 1024;
let queryImage = null; // id d'une image de la base (clic sur un resultat)
let uploadFile = null; // fichier envoye depuis l'ordinateur
let previewUrl = null;

function setStatus(message, isError = false) {
    $("status").textContent = message;
    $("status").classList.toggle("error", isError);
}

function applyMode() {
    const mode = $("mode").value;
    const hasText = mode === "text" || mode === "combined";
    const hasImage = mode === "image" || mode === "combined";
    $("textbox").hidden = !hasText;
    $("imagebox").hidden = !hasImage;
    $("descs").hidden = !hasImage;
    $("opwrap").hidden = mode !== "combined";
}

function applyDataset() {
    const mirflickr = $("dataset").value === "mirflickr";
    const textModes = Array.from($("mode").options).filter(
        (option) => option.value === "text" || option.value === "combined"
    );
    for (const option of textModes) {
        option.disabled = mirflickr;
    }
    $("text-support").textContent = mirflickr
        ? "MIRFLICKR ne contient pas d'annotations textuelles. Choisissez Objects pour rechercher par mots-clés."
        : "La recherche par mots-clés est disponible pour Objects.";
    if (mirflickr && $("mode").value !== "image") {
        $("mode").value = "image";
        applyMode();
        setStatus("La recherche textuelle n'est pas disponible pour MIRFLICKR. Choisissez Objects.", true);
    }
}

function showQuery(src, label) {
    $("qimg").src = src;
    $("qname").textContent = label;
    $("query").hidden = false;
    $("drop").hidden = true;
}

function clearQuery() {
    queryImage = null;
    uploadFile = null;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    $("file").value = "";
    $("query").hidden = true;
    $("drop").hidden = false;
}

function handleFile(file) {
    if (!file) return;
    if (!["image/png", "image/jpeg"].includes(file.type)) {
        setStatus("Format non accepte : utilisez PNG ou JPEG.", true);
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
            showQuery(hit.url, hit.category || hit.id);
            // Basculer en mode image si on etait en texte
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
    const k = Number($("k").value) || 20;
    const dataset = $("dataset").value;
    const text = mode === "image" ? "" : $("text").value.trim();
    const descriptors = Array.from(
        document.querySelectorAll("#descs input:checked"),
        (input) => input.value
    );
    const operator = $("operator").value;

    if (mode === "text" && !text) {
        return setStatus("Saisissez des mots-cles.", true);
    }
    if (mode === "combined" && !text) {
        return setStatus("Saisissez des mots-cles pour la recherche combinee.", true);
    }
    if (mode !== "text" && !queryImage && !uploadFile) {
        return setStatus("Choisissez une image : upload ou clic sur un resultat.", true);
    }
    if (mode !== "text" && descriptors.length === 0) {
        return setStatus("Selectionnez au moins un descripteur.", true);
    }

    if (mode === "text") {
        // Recherche textuelle
        setStatus("Recherche en cours...");
        try {
            const res = await fetch("/search", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ dataset, text, k }),
            });
            const data = await res.json();
            if (!res.ok) {
                render([]);
                return setStatus(typeof data.detail === "string" ? data.detail : "Erreur.", true);
            }
            render(data.results);
            setStatus(data.count + " resultat(s)");
        } catch (err) {
            render([]);
            setStatus("Impossible de joindre l'API. Est-elle demarree ?", true);
        }

    } else if (uploadFile) {
        // Image envoyee depuis l'ordinateur, avec ou sans mots-cles
        setStatus("Recherche en cours...");
        try {
            const fd = new FormData();
            fd.append("file", uploadFile);
            fd.append("dataset", dataset);
            fd.append("k", String(k));
            fd.append("descriptors", descriptors.join(","));
            fd.append("text", text);
            fd.append("operator", operator);
            const res = await fetch("/search/upload", { method: "POST", body: fd });
            const data = await res.json();
            if (!res.ok) {
                render([]);
                return setStatus(typeof data.detail === "string" ? data.detail : "Erreur.", true);
            }
            render(data.results);
            setStatus(data.count + " resultat(s)");
        } catch (err) {
            render([]);
            setStatus("Impossible de joindre l'API. Est-elle demarree ?", true);
        }
    } else {
        // Image de la base, avec ou sans mots-cles
        setStatus("Recherche en cours...");
        try {
            const res = await fetch("/search", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    dataset,
                    text,
                    query_image_id: queryImage,
                    descriptors,
                    operator,
                    k,
                }),
            });
            const data = await res.json();
            if (!res.ok) {
                render([]);
                return setStatus(typeof data.detail === "string" ? data.detail : "Erreur.", true);
            }
            render(data.results);
            setStatus(data.count + " resultat(s)");
        } catch (err) {
            render([]);
            setStatus("Impossible de joindre l'API. Est-elle demarree ?", true);
        }
    }
}

// Event listeners
$("mode").addEventListener("change", applyMode);
$("dataset").addEventListener("change", applyDataset);
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
applyDataset();