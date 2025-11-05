def energy_tip(energy):
    if energy > 4:
        return "⚠️ High usage! Use energy-efficient appliances and avoid peak hours."
    elif energy > 2:
        return "🙂 Moderate usage. Switch off unnecessary lights and fans."
    else:
        return "✅ Low usage! Keep it up and continue saving energy."
