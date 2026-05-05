import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# 1. Cargar datos y hacer downsample
df = pd.read_csv(r"C:\Users\IgnacioGonzalez\Desktop\Proyectos\TFM-Logistics-Baseline\data\animation_timeline_s211_fifo.csv")
df_sub = df.iloc[::50].reset_index(drop=True)

# 2. Configuración de estilo "Premium"
plt.style.use('seaborn-v0_8-whitegrid')  # Estilo base con grid limpio
COLOR_NORMAL = '#2980b9'  # Azul corporativo
COLOR_URGENTE = '#e74c3c'  # Rojo alerta (para captar la atención)
COLOR_FONDO = '#f4f7f6'  # Gris muy claro para la figura completa
COLOR_TEXTO = '#2c3e50'  # Gris oscuro para leer mejor que el negro puro

# Creamos la figura un poco más ancha para que quepa el panel de KPIs
fig, ax = plt.subplots(figsize=(11, 6), facecolor=COLOR_FONDO)
plt.subplots_adjust(right=0.65, left=0.1)
fases = ['Picking', 'Packing', 'Dispatch']
x = np.arange(len(fases))
width = 0.55

# Calcular límite Y con un 15% de margen superior para que respire
max_queue = max(
    (df['pick_queue_urgent'] + df['pick_queue_normal']).max(),
    (df['pack_queue_urgent'] + df['pack_queue_normal']).max(),
    (df['dispatch_queue_urgent'] + df['dispatch_queue_normal']).max()
)
y_max = max_queue + (max_queue * 0.15)


def animar(i):
    ax.clear()

    # Restaurar limpieza del eje tras el clear()
    ax.set_facecolor(COLOR_FONDO)
    ax.spines['top'].set_visible(False)  # Quitar borde superior
    ax.spines['right'].set_visible(False)  # Quitar borde derecho
    ax.spines['left'].set_color('#bdc3c7')
    ax.spines['bottom'].set_color('#bdc3c7')
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)  # Solo líneas horizontales suaves

    row = df_sub.iloc[i]

    urgentes = [row['pick_queue_urgent'], row['pack_queue_urgent'], row['dispatch_queue_urgent']]
    normales = [row['pick_queue_normal'], row['pack_queue_normal'], row['dispatch_queue_normal']]

    # Dibujar barras con borde blanco para separar perfectamente los colores
    p1 = ax.bar(x, normales, width, color=COLOR_NORMAL, label='Pedidos Normales', edgecolor='white', linewidth=1.5,
                zorder=3)
    p2 = ax.bar(x, urgentes, width, bottom=normales, color=COLOR_URGENTE, label='Pedidos Urgentes', edgecolor='white',
                linewidth=1.5, zorder=3)

    # Configuración de Ejes y Textos
    ax.set_ylim(0, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels(fases, fontsize=12, fontweight='bold', color=COLOR_TEXTO)
    ax.set_ylabel("Volumen de Pedidos en Cola", fontsize=11, fontweight='bold', color=COLOR_TEXTO)

    # Título principal y Subtítulo de tiempo centrados y limpios
    ax.text(0.5, 1.08, "Real Time Queue Evolution - FIFO Policy",
            horizontalalignment='center', fontsize=16, fontweight='bold', color=COLOR_TEXTO, transform=ax.transAxes)
    ax.text(0.5, 1.02, f"s211 Scenario | Simulation Minute: {row['time_min']:.0f}",
            horizontalalignment='center', fontsize=12, color='#7f8c8d', transform=ax.transAxes)

    # Leyenda limpia sin bordes oscuros
    ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none', fontsize=10)

    # Poner el número total con una fuente clara y un poco por encima de la barra
    for j in range(3):
        total = normales[j] + urgentes[j]
        if total > 0:
            ax.text(x[j], total + (y_max * 0.02), str(int(total)), ha='center', va='bottom', fontsize=11,
                    fontweight='bold', color=COLOR_TEXTO)

    # --- PANEL LATERAL DE KPIs EN VIVO ---
    kpi_text = (
        f"System State\n"
        f"------------------------\n"
        f"Occupied workers:\n"
        f" • Picking: {int(row['picking_busy'])}\n"
        f" • Packing: {int(row['packing_busy'])}\n"
        f" • Dispatch: {int(row['dispatch_busy'])}\n\n"
        f"Total Progress:\n"
        f" • In System: {int(row['in_system_total'])}\n"
        f" • Completed: {int(row['completed_total'])}"
    )
    # Dibujar la caja de KPIs a la derecha del gráfico
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, edgecolor='#bdc3c7')
    ax.text(1.03, 0.5, kpi_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', bbox=props, color=COLOR_TEXTO, family='monospace')


# Usamos bbox_inches='tight' al guardar para que no corte el panel de KPIs
# Aumentamos un poco los FPS (15) y bajamos el interval para que sea más fluido
ani = animation.FuncAnimation(fig, animar, frames=len(df_sub), interval=80)
ani.save('evolucion_s211_fifo_pro.gif', writer='pillow', fps=15, savefig_kwargs={'bbox_inches': 'tight'})
print("¡GIF profesional generado con éxito!")