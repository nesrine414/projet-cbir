// Client web : appelle l'API (POST /search) et affiche la grille de résultats.
const $ = (id) => document.getElementById(id);
let queryImage = null; // id de l'image requête choisie (ex. "butterfly/image_0001.jpg")

function setStatus(message, isError = false) {
    const el = $("status");
    el.textContent = message;
    el.classList.toggle("error", isError);
}

function setQueryImage(id, url) {
    queryImage = id;
    $("query").hidden = id === null;
    if (id !== null) {
        $("qimg").src = url;
        $("qid").textContent = id;
    }
}

function render(results) {
    const grid = $("results");
    grid.replaceChildren();
    for (const hit of results) {
        const card = document.createElement("figure");
        card.className = "card";
        card.style.margin = "0";

        const btn = document.createElement("button");
        btn.type = "button";
        btn.title = "Rechercher les images similaires";
        const img = document.createElement("img");
        img.src = hit.url;
        img.alt = hit.id;
        img.loading = "lazy";
        btn.appendChild(img);
        btn.addEventListener("click", () => {
            setQueryImage(hit.id, hit.url);
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
    const text = $("text").value.trim();
    if (!text && !queryImage) {
        setStatus("Saisissez des mots-clés ou cliquez sur une image.", true);
        return;
    }
    const body = {
        dataset: $("dataset").value,
        text: text || null,
        query_image_id: queryImage,
        descriptors: [...document.querySelectorAll("#descs input:checked")].map((c) => c.value),
        operator: $("operator").value,
        k: Number($("k").value) || 20,
    };
    setStatus("Recherche en cours…");
    try {
        const res = await fetch("/search", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        const data = await res.json();
        if (!res.ok) {
            render([]);
            setStatus(typeof data.detail === "string" ? data.detail : "Requête invalide.", true);
            return;
        }
        render(data.results);
        setStatus(data.count + " résultat(s)");
    } catch (err) {
        render([]);
        setStatus("Impossible de joindre l'API. Est-elle démarrée ?", true);
    }
}

$("form").addEventListener("submit", (e) => {
    e.preventDefault();
    runSearch();
});
$("clear").addEventListener("click", () => setQueryImage(null));