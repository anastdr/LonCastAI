const form = document.querySelector("#searchForm");
const postcodeInput = document.querySelector("#postcodeInput");
const queryInput = document.querySelector("#queryInput");
const errorBox = document.querySelector("#errorBox");
const successBox = document.querySelector("#successBox");
const autocompleteResults = document.querySelector("#autocompleteResults");
const suggestionsSection = document.querySelector("#suggestionsSection");
const suggestionsList = document.querySelector("#suggestionsList");
const mapSection = document.querySelector("#mapSection");
const emptyState = document.querySelector("#emptyState");
const estimatePanel = document.querySelector("#estimatePanel");
const selectedAddress = document.querySelector("#selectedAddress");
const mainPrediction = document.querySelector("#mainPrediction");
const lastSoldPanel = document.querySelector("#lastSoldPanel");
const baselinePrediction = document.querySelector("#baselinePrediction");
const mlPrediction = document.querySelector("#mlPrediction");
const mixedPrediction = document.querySelector("#mixedPrediction");
const explanationList = document.querySelector("#explanationList");
const detailsToggle = document.querySelector("#detailsToggle");
const saveSearchButton = document.querySelector("#saveSearchButton");
const confidenceMarker = document.querySelector(".confidence-meter span");
const mapStatusTitle = document.querySelector("#mapStatusTitle");
const mapStatusText = document.querySelector("#mapStatusText");
const explanationModal = document.querySelector("#explanationModal");
const modalClose = document.querySelector("#modalClose");
const modalAddress = document.querySelector("#modalAddress");
const modalExplanationList = document.querySelector("#modalExplanationList");
const contactForm = document.querySelector("#contactForm");
const contactStatus = document.querySelector("#contactStatus");

let currentPostcode = "";
let currentEstimate = null;
let propertyMap = null;
let propertyMarker = null;
let postcodeCircle = null;
let amenityLayer = null;
let autocompleteTimer = null;
let latestAutocompleteRequest = 0;

function applyTimeTheme() {
  const hour = new Date().getHours();
  const theme = hour >= 7 && hour < 19 ? "light" : "dark";
  document.documentElement.dataset.theme = theme;
}

function formatCurrency(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "Not available";
  }
  return new Intl.NumberFormat("en-GB", {
    style: "currency",
    currency: "GBP",
    maximumFractionDigits: 0,
  }).format(Number(value));
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function showMessage(type, title, body) {
  const box = type === "error" ? errorBox : successBox;
  const other = type === "error" ? successBox : errorBox;
  other.hidden = true;
  box.innerHTML = `<strong>${title}</strong><br>${body}`;
  box.hidden = false;
}

function clearMessages() {
  errorBox.hidden = true;
  successBox.hidden = true;
}

async function initMap() {
  if (!window.L) {
    mapStatusTitle.textContent = "Map unavailable";
    mapStatusText.textContent = "The interactive map needs internet access to load map tiles.";
    return;
  }

  const mapConfig = await getJson("/map-config").catch(() => ({
    provider: "fallback",
  }));
  const hasOsMaps = Boolean(mapConfig.os_maps_api_key);
  const osLayer = mapConfig.os_maps_layer || "Light_3857";
  const tileUrl = hasOsMaps
    ? `https://api.os.uk/maps/raster/v1/zxy/${encodeURIComponent(osLayer)}/{z}/{x}/{y}.png?key=${encodeURIComponent(mapConfig.os_maps_api_key)}`
    : "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png";
  const tileOptions = hasOsMaps
    ? {
        maxZoom: 20,
        attribution: "Contains OS data &copy; Crown copyright and database rights",
      }
    : {
        maxZoom: 19,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
      };

  propertyMap = L.map("propertyMap", {
    scrollWheelZoom: false,
    zoomControl: true,
  }).setView([51.5074, -0.1278], 11);

  L.tileLayer(tileUrl, tileOptions).addTo(propertyMap);
  amenityLayer = L.layerGroup().addTo(propertyMap);
  mapStatusTitle.textContent = hasOsMaps ? "OS Maps ready" : "London map";
  mapStatusText.textContent = hasOsMaps
    ? "Using Ordnance Survey OS Maps API. Search a postcode to focus the map."
    : "Using a fallback web map. Add an OS Maps API key for Ordnance Survey mapping.";

  requestAnimationFrame(() => propertyMap.invalidateSize(false));
  setTimeout(() => propertyMap.invalidateSize(false), 200);
  setTimeout(() => propertyMap.invalidateSize(false), 650);
  window.addEventListener("resize", () => propertyMap.invalidateSize());
}

function updateMapFromEstimate(estimate) {
  if (!propertyMap || !estimate) return;

  const features = estimate.features || {};
  const lat = Number(features.latitude);
  const lon = Number(features.longitude);
  const postcode = features.postcode || currentPostcode || "selected postcode";

  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    propertyMap.setView([51.5074, -0.1278], 11);
    mapStatusTitle.textContent = "Location not available";
    mapStatusText.textContent = "This property does not have coordinates in the loaded dataset yet.";
    return;
  }

  const position = [lat, lon];
  const markerIcon = L.divIcon({
    className: "",
    html: '<span class="property-marker">LV</span>',
    iconSize: [34, 34],
    iconAnchor: [17, 17],
    popupAnchor: [0, -18],
  });

  if (propertyMarker) propertyMarker.remove();
  if (postcodeCircle) postcodeCircle.remove();
  if (amenityLayer) amenityLayer.clearLayers();

  propertyMarker = L.marker(position, { icon: markerIcon })
    .addTo(propertyMap)
    .bindPopup(`<strong>${escapeHtml(postcode)}</strong><br>${escapeHtml(estimate.full_address || "Selected property")}`);

  postcodeCircle = L.circle(position, {
    radius: 420,
    color: "#2f5f7f",
    fillColor: "#6b8faa",
    fillOpacity: 0.16,
    weight: 2,
  }).addTo(propertyMap);

  const amenityBounds = [position];
  const amenities = Array.isArray(features.map_amenities) ? features.map_amenities : [];
  amenities.forEach((amenity) => {
    const amenityLat = Number(amenity.latitude);
    const amenityLon = Number(amenity.longitude);
    if (!Number.isFinite(amenityLat) || !Number.isFinite(amenityLon) || !amenityLayer) {
      return;
    }

    const amenityType = amenity.type === "station" ? "station" : "school";
    const amenityPosition = [amenityLat, amenityLon];
    const amenityIcon = L.divIcon({
      className: "",
      html: `<span class="amenity-marker amenity-${amenityType}">${amenityType === "station" ? "T" : "S"}</span>`,
      iconSize: [30, 30],
      iconAnchor: [15, 15],
      popupAnchor: [0, -16],
    });
    const distance = Number.isFinite(Number(amenity.distance_km))
      ? `${Number(amenity.distance_km).toFixed(2)} km`
      : "distance unavailable";
    const detail = amenity.detail ? `<br>${escapeHtml(amenity.detail)}` : "";

    L.marker(amenityPosition, { icon: amenityIcon })
      .addTo(amenityLayer)
      .bindPopup(`<strong>${escapeHtml(amenity.label || "Nearby amenity")}</strong><br>${escapeHtml(amenity.name)}<br>${distance}${detail}`);

    L.polyline([position, amenityPosition], {
      color: amenityType === "station" ? "#2f5f7f" : "#5e956b",
      weight: 2,
      opacity: 0.72,
      dashArray: "5 7",
    }).addTo(amenityLayer);

    amenityBounds.push(amenityPosition);
  });

  if (amenityBounds.length > 1) {
    propertyMap.fitBounds(amenityBounds, {
      animate: true,
      maxZoom: 16,
      padding: [48, 48],
    });
  } else {
    propertyMap.setView(position, 15, { animate: true });
  }
  propertyMarker.openPopup();
  mapStatusTitle.textContent = postcode;
  mapStatusText.textContent = amenities.length
    ? "Showing selected property plus closest station and school."
    : estimate.full_address || "Focused on the selected property area.";
}

function validatePostcode(postcode) {
  const cleaned = normalizePostcode(postcode);
  if (!cleaned) {
    return "Please enter a postcode before searching.";
  }
  if (cleaned.length < 5) {
    return "The postcode looks too short. Enter a full London postcode such as SW6 2HU.";
  }
  if (!/[A-Z]{1,2}\d/.test(cleaned)) {
    return "The postcode format is not recognised. Use letters and numbers, for example W14 0TL.";
  }
  return "";
}

function normalizePostcode(postcode) {
  return String(postcode || "").trim().toUpperCase().replace(/\s+/g, "");
}

function formatPostcode(postcode) {
  const cleaned = normalizePostcode(postcode);
  if (cleaned.length <= 3) return cleaned;
  return `${cleaned.slice(0, -3)} ${cleaned.slice(-3)}`;
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`The server returned ${response.status}. Please try again.`);
  }
  return response.json();
}

function showContactStatus(type, message) {
  contactStatus.className = `contact-status ${type}`;
  contactStatus.textContent = message;
  contactStatus.hidden = false;
}

async function sendContactMessage(event) {
  event.preventDefault();
  const formData = new FormData(contactForm);
  const payload = {
    name: String(formData.get("name") || "").trim(),
    email: String(formData.get("email") || "").trim(),
    message: String(formData.get("message") || "").trim(),
  };

  if (!payload.name || !payload.email || payload.message.length < 10) {
    showContactStatus("error", "Please enter your name, a valid email, and a message with at least 10 characters.");
    return;
  }

  const submitButton = contactForm.querySelector("button[type='submit']");
  submitButton.disabled = true;
  showContactStatus("success", "Sending your message...");

  try {
    const response = await fetch("/contact", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok || result.success === false) {
      throw new Error(result.detail || result.message || "The message could not be sent.");
    }
    contactForm.reset();
    showContactStatus("success", result.message || "Thank you. Your message has been sent.");
  } catch (error) {
    showContactStatus("error", error.message || "The message could not be sent. Please try again later.");
  } finally {
    submitButton.disabled = false;
  }
}

async function searchProperties(event) {
  event.preventDefault();
  const shouldScrollToResults = event.submitter?.matches("button[type='submit']") ?? false;
  clearMessages();
  suggestionsList.innerHTML = "";
  suggestionsSection.hidden = true;
  hideAutocomplete();

  const postcode = formatPostcode(postcodeInput.value);
  const query = queryInput.value.trim();
  const validationError = validatePostcode(postcode);

  if (validationError) {
    showMessage("error", "Check the postcode", validationError);
    postcodeInput.focus();
    return;
  }

  form.querySelector("button[type='submit']").disabled = true;
  postcodeInput.value = postcode;
  currentPostcode = postcode;

  try {
    const postcodeCheck = await getJson(`/check-postcode?postcode=${encodeURIComponent(postcode)}`);
    if (!postcodeCheck.valid_london_postcode) {
      showMessage(
        "error",
        "Postcode is outside the supported area",
        "LonCastAI currently supports London postcodes from the loaded dataset. Check the postcode spelling or try another London postcode."
      );
      return;
    }

    const properties = await getJson(
      `/property-search?postcode=${encodeURIComponent(postcode)}&query=${encodeURIComponent(query)}&limit=24`
    );

    if (!properties.length) {
      const coverage = await getJson("/data-coverage");
      const prefixes = coverage.postcode_prefixes?.join(", ") || "the currently loaded subset";
      showMessage(
        "error",
        "This postcode is not in the loaded property dataset",
        `The postcode is valid, but no property features were loaded for it. The current database covers ${prefixes}. Try one of those areas, or rebuild the data pipeline including this postcode prefix.`
      );
      return;
    }

    renderSuggestions(properties);
    showMessage("success", "Properties found", `Choose one of ${properties.length} matching addresses to generate a prediction.`);
    if (shouldScrollToResults) {
      mapSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  } catch (error) {
    showMessage("error", "Search failed", error.message || "Something went wrong while searching. Please check the backend is running.");
  } finally {
    form.querySelector("button[type='submit']").disabled = false;
  }
}

function saleSourceLabel(property) {
  return property.sale_price_source === "exact_matched_sale" ? "Exact sold-price match" : "Area fallback";
}

function renderSuggestions(properties) {
  renderAutocomplete(properties);
  suggestionsSection.hidden = true;
  suggestionsList.innerHTML = "";
}

function renderAutocomplete(properties) {
  if (!properties.length) {
    hideAutocomplete();
    return;
  }

  autocompleteResults.innerHTML = properties
    .map((property) => {
      return `
        <button class="autocomplete-option" type="button" role="option" data-id="${property.id}">
          <strong>${escapeHtml(property.full_address)}</strong>
          <span>${escapeHtml(property.property_subtype || "Property")} · EPC ${escapeHtml(property.epc_rating || "n/a")} · ${escapeHtml(property.floor_area || "?")} sqm · ${escapeHtml(saleSourceLabel(property))}</span>
        </button>
      `;
    })
    .join("");
  autocompleteResults.hidden = false;
}

function renderPostcodeAutocomplete(postcodes) {
  if (!postcodes.length) {
    hideAutocomplete();
    return;
  }

  autocompleteResults.innerHTML = postcodes
    .map((item) => {
      const postcode = item.postcode || formatPostcode(item.postcode_clean || "");
      return `
        <button class="autocomplete-option postcode-option" type="button" role="option" data-postcode="${escapeHtml(postcode)}">
          <strong>${escapeHtml(postcode)}</strong>
          <span>Loaded postcode area. Select to search matching properties.</span>
        </button>
      `;
    })
    .join("");
  autocompleteResults.hidden = false;
}

function hideAutocomplete() {
  autocompleteResults.hidden = true;
  autocompleteResults.innerHTML = "";
}

async function refreshAutocomplete() {
  const rawPostcode = postcodeInput.value.trim();
  const postcode = formatPostcode(rawPostcode);
  const query = queryInput.value.trim();
  const validationError = validatePostcode(postcode);
  const cleanedPostcode = normalizePostcode(rawPostcode);

  if (cleanedPostcode.length > 0 && validationError) {
    const requestId = ++latestAutocompleteRequest;
    try {
      const postcodes = await getJson(
        `/postcode-suggestions?query=${encodeURIComponent(cleanedPostcode)}&limit=10`
      );
      if (requestId === latestAutocompleteRequest) {
        renderPostcodeAutocomplete(postcodes);
      }
    } catch {
      if (requestId === latestAutocompleteRequest) {
        hideAutocomplete();
      }
    }
    return;
  }

  if (validationError) {
    hideAutocomplete();
    return;
  }

  const requestId = ++latestAutocompleteRequest;
  try {
    const properties = await getJson(
      `/property-search?postcode=${encodeURIComponent(postcode)}&query=${encodeURIComponent(query)}&limit=8`
    );
    if (requestId === latestAutocompleteRequest) {
      renderAutocomplete(properties);
    }
  } catch {
    if (requestId === latestAutocompleteRequest) {
      hideAutocomplete();
    }
  }
}

function scheduleAutocomplete() {
  clearTimeout(autocompleteTimer);
  autocompleteTimer = setTimeout(refreshAutocomplete, 260);
}

async function loadEstimate(propertyId) {
  clearMessages();
  try {
    const estimate = await getJson(`/estimate?property_id=${encodeURIComponent(propertyId)}`);
    if (estimate.error) {
      showMessage("error", "Prediction unavailable", estimate.error);
      return;
    }
    currentEstimate = estimate;
    currentPostcode = estimate.features?.postcode || postcodeInput.value.trim();
    queryInput.value = estimate.full_address || queryInput.value;
    renderEstimate();
    updateMapFromEstimate(estimate);
    hideAutocomplete();
  } catch (error) {
    showMessage("error", "Prediction failed", error.message || "The estimate could not be loaded.");
  }
}

function renderEstimate() {
  if (!currentEstimate) return;

  const summary = currentEstimate.prediction_summary || {};
  const ml = currentEstimate.ml_predictions || {};
  const features = currentEstimate.features || {};
  const confidence = ml.confidence_score || 0.35;
  const explanations = currentEstimate.explanations || [];
  const visibleExplanations = explanations.slice(0, 5);

  emptyState.hidden = true;
  estimatePanel.hidden = false;
  selectedAddress.textContent = currentEstimate.full_address;
  mainPrediction.textContent = formatCurrency(currentEstimate.mixed_prediction || currentEstimate.estimated_price);
  if (features.last_sold_price) {
    const transferDate = features.last_transfer_date ? `Sold ${escapeHtml(features.last_transfer_date.slice(0, 10))}` : "";
    const indexedPrice = features.indexed_last_sold_price
      ? `<span>HPI-adjusted: ${formatCurrency(features.indexed_last_sold_price)}</span>`
      : "";
    lastSoldPanel.innerHTML = `
      <span>Exact last sold</span>
      <strong>${formatCurrency(features.last_sold_price)}</strong>
      <em>${transferDate}</em>
      ${indexedPrice}
    `;
  } else {
    lastSoldPanel.innerHTML = `
      <span>Last sold</span>
      <strong>Not available</strong>
      <em>No safe exact sale match in the loaded price-paid data.</em>
    `;
  }
  baselinePrediction.textContent = formatCurrency(summary.baseline_prediction || currentEstimate.base_estimator_prediction);
  mlPrediction.textContent = ml.available ? formatCurrency(summary.ml_prediction || ml.blended_ml_prediction) : "Train model";
  mixedPrediction.textContent = formatCurrency(summary.mixed_prediction || currentEstimate.mixed_prediction);
  confidenceMarker.style.width = `${Math.max(8, Math.min(95, confidence * 100))}%`;
  explanationList.innerHTML = visibleExplanations.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
  detailsToggle.textContent = "See full explanation";
}

function openExplanationModal() {
  if (!currentEstimate) {
    showMessage("error", "No prediction yet", "Search for a postcode and choose a property before opening the full explanation.");
    return;
  }

  const explanations = currentEstimate.explanations || [];
  modalAddress.textContent = currentEstimate.full_address || "";
  modalExplanationList.innerHTML = explanations.length
    ? explanations.map((item) => `<li>${escapeHtml(item)}</li>`).join("")
    : "<li>No detailed explanation is available for this property yet.</li>";
  explanationModal.hidden = false;
  document.body.classList.add("modal-open");
  modalClose.focus();
}

function closeExplanationModal() {
  explanationModal.hidden = true;
  document.body.classList.remove("modal-open");
  detailsToggle.focus();
}

function saveCurrentSearch() {
  if (!currentPostcode) {
    showMessage("error", "Nothing to save", "Search for a postcode first, then save it for later.");
    return;
  }
  const saved = JSON.parse(localStorage.getItem("loncastai:saved-searches") || "[]");
  const nextItem = { postcode: currentPostcode, query: queryInput.value.trim(), savedAt: new Date().toISOString() };
  localStorage.setItem("loncastai:saved-searches", JSON.stringify([nextItem, ...saved].slice(0, 10)));
  showMessage("success", "Search saved", `${currentPostcode.toUpperCase()} has been saved in this browser.`);
}

applyTimeTheme();
initMap();
form.addEventListener("submit", searchProperties);
contactForm.addEventListener("submit", sendContactMessage);
saveSearchButton.addEventListener("click", saveCurrentSearch);
detailsToggle.addEventListener("click", openExplanationModal);
postcodeInput.addEventListener("input", scheduleAutocomplete);
postcodeInput.addEventListener("blur", () => {
  postcodeInput.value = formatPostcode(postcodeInput.value);
});
queryInput.addEventListener("input", scheduleAutocomplete);
autocompleteResults.addEventListener("click", (event) => {
  const option = event.target.closest("[data-id]");
  if (option) {
    loadEstimate(option.dataset.id);
    return;
  }
  const postcodeOption = event.target.closest("[data-postcode]");
  if (postcodeOption) {
    postcodeInput.value = postcodeOption.dataset.postcode;
    queryInput.value = "";
    refreshAutocomplete();
  }
});
suggestionsList.addEventListener("click", (event) => {
  const card = event.target.closest("[data-id]");
  if (card) {
    loadEstimate(card.dataset.id);
  }
});
modalClose.addEventListener("click", closeExplanationModal);
explanationModal.addEventListener("click", (event) => {
  if (event.target === explanationModal) {
    closeExplanationModal();
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !explanationModal.hidden) {
    closeExplanationModal();
  }
});
document.addEventListener("click", (event) => {
  if (!form.contains(event.target)) {
    hideAutocomplete();
  }
});
