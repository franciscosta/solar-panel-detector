import streamlit as st
from ultralytics import YOLO
from PIL import Image
import random
import requests
import folium
from geopy.geocoders import Nominatim
import overpy
from streamlit_folium import st_folium
from streamlit_drawable_canvas import st_canvas
import numpy as np

#Load pre trained model
model = YOLO("detector.pt")

#Panel detection logic
def plot_results(im_array):
    return Image.fromarray(im_array[..., ::-1])

def solar_panel_predict(image, conf=0.7, mask_color=(0, 255, 0)):

    negative_sentences = [
        "No solar panels yet? Your roof is a blank canvas waiting for a green masterpiece! 🎨🌱",
        "It's lonely up here without solar panels. 🌞🎉",
        "Your roof could be a superhero in disguise. Just needs its solar cape! 🦸‍♂️☀️",
        "Clear skies, empty roof. It's the perfect opportunity to harness the sun! 🌤️🔋"
    ]
    positive_sentences = [
        "Solar panels detected! 🌍💚",
        "Sunny side up! ☀️😎",
        "You're a Renewable Energy Rockstar! ⭐🌱"
    ]

    results = model(image, stream=True, conf=conf)

    for result in results:
        img = np.array(image.convert("RGB")) 
        panel_detected = False

        if hasattr(result, "masks") and result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            for mask in masks:
                panel_detected = True
                mask_resized = np.array(Image.fromarray(mask).resize(
                    (img.shape[1], img.shape[0]), resample=Image.NEAREST))
                binary_mask = mask_resized.astype(bool)
                img[binary_mask] = mask_color
        else:
            confs = result.boxes.conf.numpy().tolist() if result.boxes is not None else []
            panel_detected = bool(confs)

        message = random.choice(positive_sentences if panel_detected else negative_sentences)
        return Image.fromarray(img), message

#Convert address to coordinates
def get_coordinates(address):
    geolocator = Nominatim(user_agent="solar_locator")
    location = geolocator.geocode(address)
    if not location:
        raise Exception(f"Address '{address}' not found.")
    return location.latitude, location.longitude

#Find nearby shops
def find_solar_services(lat, lon, radius=15000):
    api = overpy.Overpass()
    api.timeout = 25  # Set a safe timeout
    query = f"""
    (
    node["shop"="solar"](around:{radius},{lat},{lon}); #Avoid generic name filters like 'name~solar' which return unrelated businesses
    node["office"="energy"](around:{radius},{lat},{lon});
    node["craft"="electrician"]["description"~"solar", i](around:{radius},{lat},{lon});
    node["generator:source"="solar"](around:{radius},{lat},{lon});
    );
    out;
    """
    try:
        result = api.query(query)
        services = [(n.tags.get("name", "Unnamed"), n.lat, n.lon) for n in result.nodes]
        if services:
            return services, True
    except Exception:
        pass

    #Add services with real coordinates
    fallback_services = [
        ("Svestsolar", 38.697, -9.318),
        ("Solarassist", 38.688, -9.329),
        ("Energia Solar Solutions", 38.708, -9.336),
        ("Smart EV Chargers & Energy", 38.680, -9.325),
        ("Solarimpact", 38.787, -9.134),
        ("Solar Planet Portugal", 38.755, -9.226),
        ("Proposta Renovável", 38.630, -9.084),
        ("Greenlevel - Soluções e Sistemas", 38.773, -9.145),
        ("Da Silva Instalações Elétricas", 38.662, -9.098),
        ("Solar Mais - Energia e Ambiente", 38.671, -9.198),
        ("Electricplace Unipessoal", 38.749, -9.175),
        ("Solcor Portugal", 38.715, -9.140),
        ("SunEnergy", 38.688, -9.320)
    ]

    #Filter results by distance
    from geopy.distance import geodesic
    filtered = [
        (name, lat_, lon_) for name, lat_, lon_ in fallback_services
        if geodesic((lat, lon), (lat_, lon_)).meters <= radius
    ]
    return filtered, False


#Create the map
def create_map(lat, lon, services):
    m = folium.Map(location=[lat, lon], zoom_start=14)
    folium.Marker([lat, lon], popup="Your Location", icon=folium.Icon(color="blue")).add_to(m)
    for name, s_lat, s_lon in services:
        folium.Marker([s_lat, s_lon], popup=name, icon=folium.Icon(color="green")).add_to(m)
    return m

# Setup streamlit app
st.set_page_config(
    page_title="HuggingSUN",
    page_icon="HuggingSUN.png",
    layout="wide"
)

# Load logos
huggingsun_logo = Image.open("HuggingSUN.png")  # Place HuggingSUN.png in the same directory
nova_logo = Image.open("nova_logo.png")  # Place nova_logo.png in the same directory

# Create three columns: HuggingSUN logo | Title | Nova logo
col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    st.image(huggingsun_logo, width=200)

with col2:
    st.markdown("""
        <div style='text-align: center;'>
            <h1 style='font-size: 46px; margin-bottom: 0;'>HuggingSUN</h1>
            <h3 style='color: gray; margin-top: 0px;'>
                Solar Panel Detection, Nearby Services Locator and Maintenance Estimator
            </h3>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.image(nova_logo, width=150)



#Image upload and detection
address = st.text_input("Enter the address of the building:")

import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import math

# Set your Google Maps API key as an environment variable or in Streamlit secrets
# Option 1 (local): set GOOGLE_MAPS_API_KEY in your environment
# Option 2 (Streamlit Cloud): add it to .streamlit/secrets.toml
import os
API_KEY = os.environ.get('GOOGLE_MAPS_API_KEY', st.secrets.get('GOOGLE_MAPS_API_KEY', ''))
if not API_KEY:
    st.error('Google Maps API key not found. Please set the GOOGLE_MAPS_API_KEY environment variable.')
    st.stop()

def geocode_address(address):
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "key": API_KEY}
    response = requests.get(url, params=params).json()
    if response["status"] != "OK":
        st.error("Address geocoding failed.")
        st.stop()
    location = response["results"][0]["geometry"]["location"]
    return location["lat"], location["lng"]

def fetch_satellite_image(lat, lng, zoom=18, size=(640, 640)):
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lng}",
        "zoom": zoom,
        "size": f"{size[0]}x{size[1]}",
        "maptype": "satellite",
        "key": API_KEY
    }
    response = requests.get(base_url, params=params)
    return Image.open(BytesIO(response.content)), zoom

def calculate_image_width_meters(lat, zoom, image_width_px):
    meters_per_pixel = 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)
    return image_width_px * meters_per_pixel

if address:
    lat, lng = geocode_address(address)
    image, zoom = fetch_satellite_image(lat, lng)
    image_width_m = calculate_image_width_meters(lat, zoom, image.size[0])

    pred_image, msg = solar_panel_predict(image, mask_color=(0, 255, 0))
    pred_array = np.array(pred_image)

    # Count green pixels in marked image
    mask = np.all(pred_array == [0, 255, 0], axis=-1)
    n_pixels = np.sum(mask)

    # Calculate area
    pixel_factor = pred_array.shape[1] / image_width_m
    area_fraction = n_pixels / (pred_array.shape[0] * pred_array.shape[1])
    m2_fraction = ((pred_array.shape[0]/pixel_factor)*(pred_array.shape[1]/pixel_factor)) * area_fraction

    maintenance_cost_upper = m2_fraction * 6 #maintenance factors estimated
    maintenance_cost_lower = m2_fraction * 4 

    st.image(pred_image, caption=msg, use_container_width=True)
    st.write(f"**Estimated Solar Panel Area:** {m2_fraction:.2f} m²")
    st.write(f"**Estimated Maintenance Cost:** {maintenance_cost_lower:.2f} € to {maintenance_cost_upper:.2f} € per year")
 

#Address-based solar services
with st.expander("📍 Find Solar Services Near an Address"):
    if address: 
        radius_km = st.slider("Search radius (km)", 1, 30, 15)

        # Use a separate flag to trigger map generation
        if "map_requested" not in st.session_state:
            st.session_state["map_requested"] = False

        if st.button("Find Nearby Services"):
            st.session_state["map_requested"] = True

        if st.session_state["map_requested"]:
            with st.spinner("Locating solar panel service providers..."):
                try:
                    lat, lon = get_coordinates(address)
                    services, used_osm = find_solar_services(lat, lon, radius=radius_km * 1000)

                    if not services:
                        st.session_state["solar_map"] = None
                        st.session_state["map_message"] = "⚠️ No solar services found nearby."
                    else:
                        st.session_state["solar_map"] = create_map(lat, lon, services)
                        st.session_state["map_message"] = f"✅ Found {len(services)} result(s) from OpenStreetMap."

                except Exception as e:
                    st.session_state["solar_map"] = None
                    st.session_state["map_message"] = f"❌ Error: {e}"

    # Display what is saved in session_state
    if "solar_map" in st.session_state and st.session_state["solar_map"]:
        st.info(st.session_state["map_message"])
        
        col_map, col_list = st.columns([2, 1]) 

        with col_map:
            st_folium(
                st.session_state["solar_map"],
                width=1000,
                height=700,
                returned_objects=[],
                key="static_map"
            )

        with col_list:
            st.markdown("### 🏪 Nearby Solar Services")
            for name, s_lat, s_lon in services:
                st.markdown(f"- **{name}**\n\n📍 {round(s_lat, 4)}, {round(s_lon, 4)}")

        #Contact form
        st.markdown("---")
        st.subheader("📬 Contact a Service Provider")

        with st.form("contact_form"):
            st.write("Please fill out the form to send a request to a selected provider:")

            user_name = st.text_input("Your Name")
            user_email = st.text_input("Your Email")
            selected_provider = st.selectbox(
                "Select a Provider",
                [name for name, _, _ in services]
            )
            service_type = st.selectbox(
                "Type of Service",
                ["Cleaning", "Maintenance", "Installation", "Other"]
            )
            user_message = st.text_area("Additional Message", placeholder="Write your request or question...")

            submitted = st.form_submit_button("Send Request")

            if submitted:
                st.success(f"✅ Request sent to **{selected_provider}**! We’ll get back to you shortly.")

    elif "map_message" in st.session_state:
        st.warning(st.session_state["map_message"])