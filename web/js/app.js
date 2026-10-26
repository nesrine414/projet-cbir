// Client web : appelle l'API et affiche les résultats.
const $ = (id) => document.getElementById(id);
const MAX_BYTES = 10 * 1024 * 1024;
const state = {
    queryImage: null,
    uploadFile: null,
    previewUrl: null,
};

function setStatus(message, isError = false) {
    $("status").textContent = message;
    $("status").classList.toggle("error", isError);
}

function applyMode() {
    const mode = $("mode").value;
    $("textbox").hidden = mode === "image";
    $("imagebox").hidden = mode === "text";
    $("descs").hidden = mode === "text";
    $("operator-field").hidden = mode !== "both";
}

function showQuery(src, label) {
    $("qimg").src = src;
    $("qname").textContent = label;
    $("query").hidden = false;
    $("drop").hidden = true;
}

function clearQuery() {
    state.queryImage = null;
    state.uploadFile = null;
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = null;
    $("file").value = "";
    $("query").hidden = true;
    $("qimg").removeAttribute("src");
    $("drop").hidden = false;
}

function handleFile(file) {
    if (!file) return;
    if (!["image/png", "image/jpeg"].includes(file.type)) {
        setStatus("Format non accepté : utilisez PNG ou JPEG.", true);
        return;
    }
    if (file.size > MAX_BYTES) {
        setStatus("Image trop lourde (10 Mo maximum).", true);
        return;
    }
    clearQuery();
    state.uploadFile = file;
    state.previewUrl = URL.createObjectURL(file);
    showQuery(state.previewUrl, file.name);
    setStatus("Image prête.");
}

function selectedDescriptors() {
    const selected = Array.from(document.querySelectorAll("#descs input:checked"), (input) => input.value);
    return selected.flatMap((name) => name === "mpeg7" ? ["cld", "ehd", "scd"] : [name]);
}

async function updateDescriptorAvailability() {
    try {
        const response = await fetch("/datasets");
        if (!response.ok) return;
        const data = await response.json();
        const dataset = (data.datasets || []).find((item) => item.id === $("dataset").value);
        if (!dataset) return;
        const available = new Map((dataset.descriptors || []).map((item) => [item.id, item.available]));
        document.querySelectorAll("#descs input").forEach((input) => {
            const ids = input.value === "mpeg7" ? ["cld", "ehd", "scd"] : [input.value];
            const enabled = ids.every((id) => available.get(id));
            input.disabled = !enabled;
            input.closest("label").classList.toggle("unavailable", !enabled);
            if (!enabled) input.checked = false;
        });
    } catch (error) {
        // Keep the controls usable if dataset status cannot be fetched; search errors are shown below.
    }
}

function render(results) {
    const grid = $("results");
    grid.replaceChildren();

    for (const hit of results) {
        const card = document.createElement("figure");
        card.className = "item";

        const button = document.createElement("button");
        button.type = "button";
        button.title = "Rechercher les images similaires";
        const image = document.createElement("img");
        image.src = hit.url;
        image.alt = hit.id;
        image.loading = "lazy";
        button.appendChild(image);

        button.addEventListener("click", () => {
            if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
            state.previewUrl = null;
            state.uploadFile = null;
            $("file").value = "";
            state.queryImage = hit.id;
            showQuery(hit.url, hit.category || hit.id);
            if ($("mode").value === "text") {
                $("mode").value = "image";
                applyMode();
            }
            runSearch();
            window.scrollTo({ top: 0, behavior: "smooth" });
        });

        const info = document.createElement("div");
        const category = document.createElement("strong");
        category.textContent = hit.category || hit.id;
        const detail = document.createElement("span");
        if (hit.distance !== null && hit.distance !== undefined) {
            detail.textContent = "distance " + Number(hit.distance).toFixed(4);
        } else if (hit.score !== null && hit.score !== undefined) {
            detail.textContent = "score " + Number(hit.score).toFixed(2);
        }
        info.append(category, detail);
        card.append(button, info);
        grid.appendChild(card);
    }
}

function selectedMode() {
    return $("mode").value;
}

function errorMessage(data) {
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail)) {
        return data.detail.map((item) => item.msg || "Requête invalide").join(", ");
    }
    return "La recherche n'a pas pu aboutir.";
}

async function runSearch() {
    const mode = selectedMode();
    const k = Number($("k").value) || 18;
    const dataset = $("dataset").value;
    const text = $("text").value.trim();
    const descriptors = selectedDescriptors();
    const needsText = mode === "text" || mode === "both";
    const needsImage = mode === "image" || mode === "both";

    if (needsText && !text) {
        setStatus("Saisissez des mots-clés.", true);
        $("text").focus();
        return;
    }
    if (needsImage && !state.queryImage && !state.uploadFile) {
        setStatus("Choisissez une image requête.", true);
        return;
    }
    if (needsImage && descriptors.length === 0) {
        setStatus("Sélectionnez au moins un descripteur.", true);
        return;
    }

    const submit = $("submit");
    submit.disabled = true;
    setStatus("Recherche en cours…");

    try {
        let response;
        if (needsImage && state.uploadFile) {
            const formData = new FormData();
            formData.append("file", state.uploadFile);
            formData.append("dataset", dataset);
            formData.append("k", String(k));
            formData.append("descriptors", descriptors.join(","));
            formData.append("text", needsText ? text : "");
            formData.append("operator", $("operator").value);
            response = await fetch("/search/upload", { method: "POST", body: formData });
        } else {
            const body = { dataset, k };
            if (needsText) body.text = text;
            if (needsImage) {
                body.query_image_id = state.queryImage;
                body.descriptors = descriptors;
            }
            if (mode === "both") body.operator = $("operator").value;
            response = await fetch("/search", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
            });
        }

        const data = await response.json();
        if (!response.ok) {
            render([]);
            setStatus(errorMessage(data), true);
            return;
        }
        render(data.results || []);
        setStatus((data.count || 0) + " résultat(s)");
    } catch (error) {
        render([]);
        setStatus("Impossible de joindre l'API. Est-elle démarrée ?", true);
    } finally {
        submit.disabled = false;
    }
}

$("mode").addEventListener("change", applyMode);
$("dataset").addEventListener("change", updateDescriptorAvailability);
$("k").addEventListener("input", () => { $("kout").textContent = $("k").value; });
$("form").addEventListener("submit", (event) => {
    event.preventDefault();
    runSearch();
});
$("clear").addEventListener("click", () => {
    clearQuery();
    setStatus("");
});
$("file").addEventListener("change", (event) => handleFile(event.target.files[0]));

const drop = $("drop");
drop.addEventListener("dragover", (event) => {
    event.preventDefault();
    drop.classList.add("over");
});
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (event) => {
    event.preventDefault();
    drop.classList.remove("over");
    handleFile(event.dataTransfer.files[0]);
});

applyMode();
updateDescriptorAvailability();
