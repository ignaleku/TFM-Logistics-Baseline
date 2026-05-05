import pygame
import pandas as pd
import sys
import random


df = pd.read_csv(r"C:\Users\IgnacioGonzalez\Desktop\Proyectos\TFM-Logistics-Baseline\data\animation_timeline_s211_urgent_first.csv")
df_sub = df.iloc[::70].reset_index(drop=True)

MAX_COLA_GLOBAL = max(
    (df['pick_queue_urgent'] + df['pick_queue_normal']).max(),
    (df['pack_queue_urgent'] + df['pack_queue_normal']).max(),
    (df['dispatch_queue_urgent'] + df['dispatch_queue_normal']).max()
)
TOPE_VISUAL = int(MAX_COLA_GLOBAL * 1.1)

pygame.init()
ANCHO, ALTO = 1150, 800
pantalla = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Warehouse Flow and SCADA View")
reloj = pygame.time.Clock()


fuente_titulo = pygame.font.SysFont("Segoe UI", 22, bold=True)
fuente_subtitulo = pygame.font.SysFont("Segoe UI", 18, bold=True)
fuente_kpi = pygame.font.SysFont("Consolas", 18, bold=True)
fuente_numeros = pygame.font.SysFont("Consolas", 24, bold=True)
fuente_scada_lbl = pygame.font.SysFont("Segoe UI", 14, bold=True)


COLOR_FONDO = (25, 30, 40)
COLOR_ZONA_FISICA = (40, 45, 55)
COLOR_MESA = (100, 110, 125)
COLOR_CINTA = (15, 18, 22)
COLOR_RODILLO = (55, 65, 75)
COLOR_TRABAJADOR = (0, 200, 100)
COLOR_TEXTO = (220, 225, 230)
COLOR_NORMAL = (41, 128, 185)
COLOR_URGENTE = (192, 57, 43)
COLOR_SILO_FONDO = (30, 35, 45)
COLOR_CAJA_FISICA = (205, 133, 63)
zonas_fisicas = {
    "Picking": (50, 130, 280, 220),
    "Packing": (430, 130, 280, 220),
    "Dispatch": (810, 130, 280, 220)
}

paneles_scada = {
    "Picking": (50, 420, 280, 300),
    "Packing": (430, 420, 280, 300),
    "Dispatch": (810, 420, 280, 300)
}


class CajaMovil:
    def __init__(self, x_inicio, y_inicio, target_x):
        self.x = x_inicio
        self.y = y_inicio + random.randint(-10, 10)
        self.target_x = target_x
        self.speed = 3 + random.uniform(0, 1)
        self.size = 10
        self.activa = True

    def update(self):
        self.x += self.speed
        if self.x >= self.target_x:
            self.activa = False

    def draw(self, surface):
        if self.activa:
            pygame.draw.rect(surface, COLOR_CAJA_FISICA, (self.x, self.y, self.size, self.size), border_radius=2)
            pygame.draw.rect(surface, (100, 70, 30), (self.x, self.y, self.size, self.size), 1, border_radius=2)

todas_las_cajas_fisicas = []

def dibujar_silos_capacidad(x_panel, y_panel, normales, urgentes):
    ancho_silo = 80
    alto_silo = 160
    y_base = y_panel + alto_silo + 80


    x_normal = x_panel + 40
    pct_normal = min(normales / TOPE_VISUAL, 1.0) if TOPE_VISUAL > 0 else 0
    altura_fill_normal = int(alto_silo * pct_normal)

    pygame.draw.rect(pantalla, COLOR_SILO_FONDO, (x_normal, y_base - alto_silo, ancho_silo, alto_silo), border_radius=4)
    if altura_fill_normal > 0:
        rect_fill = (x_normal, y_base - altura_fill_normal, ancho_silo, altura_fill_normal)
        pygame.draw.rect(pantalla, COLOR_NORMAL, rect_fill, border_radius=4)
        for i in range(y_base - altura_fill_normal, y_base, 8):
            pygame.draw.line(pantalla, (20, 60, 100), (x_normal, i), (x_normal + ancho_silo, i), 1)

    pygame.draw.rect(pantalla, (100, 110, 120), (x_normal, y_base - alto_silo, ancho_silo, alto_silo), 2, border_radius=4)
    txt_n = fuente_numeros.render(str(int(normales)), True, COLOR_TEXTO if normales > 0 else (100, 100, 100))
    pantalla.blit(txt_n, (x_normal + (ancho_silo // 2) - (txt_n.get_width() // 2), y_base - alto_silo - 35))
    txt_lbl_n = fuente_scada_lbl.render("NORMAL", True, COLOR_NORMAL)
    pantalla.blit(txt_lbl_n, (x_normal + (ancho_silo // 2) - (txt_lbl_n.get_width() // 2), y_base + 5))


    x_urgente = x_panel + 160
    pct_urgente = min(urgentes / TOPE_VISUAL, 1.0) if TOPE_VISUAL > 0 else 0
    altura_fill_urgente = int(alto_silo * pct_urgente)

    pygame.draw.rect(pantalla, COLOR_SILO_FONDO, (x_urgente, y_base - alto_silo, ancho_silo, alto_silo), border_radius=4)
    if altura_fill_urgente > 0:
        rect_fill_u = (x_urgente, y_base - altura_fill_urgente, ancho_silo, altura_fill_urgente)
        pygame.draw.rect(pantalla, COLOR_URGENTE, rect_fill_u, border_radius=4)
        for i in range(y_base - altura_fill_urgente, y_base, 8):
            pygame.draw.line(pantalla, (100, 20, 20), (x_urgente, i), (x_urgente + ancho_silo, i), 1)

    pygame.draw.rect(pantalla, (100, 110, 120), (x_urgente, y_base - alto_silo, ancho_silo, alto_silo), 2, border_radius=4)
    txt_u = fuente_numeros.render(str(int(urgentes)), True, COLOR_TEXTO if urgentes > 0 else (100, 100, 100))
    pantalla.blit(txt_u, (x_urgente + (ancho_silo // 2) - (txt_u.get_width() // 2), y_base - alto_silo - 35))
    txt_lbl_u = fuente_scada_lbl.render("URGENT", True, COLOR_URGENTE)  # TRADUCIDO
    pantalla.blit(txt_lbl_u, (x_urgente + (ancho_silo // 2) - (txt_lbl_u.get_width() // 2), y_base + 5))

def dibujar_cinta_transportadora(x, y, ancho, offset_animacion):
    pygame.draw.rect(pantalla, COLOR_CINTA, (x, y + 90, ancho, 40), border_radius=5)
    pygame.draw.rect(pantalla, (10, 10, 10), (x, y + 90, ancho, 40), 2, border_radius=5)
    espacio_rodillos = 20
    for pos_x in range(0, ancho, espacio_rodillos):
        rodillo_x = x + ((pos_x + offset_animacion) % ancho)
        if x < rodillo_x < x + ancho - 5:
            pygame.draw.line(pantalla, COLOR_RODILLO, (rodillo_x, y + 92), (rodillo_x, y + 128), 3)

def dibujar_puesto_trabajo(x_fisica, y_fisica, ancho_zona, ocupados):
    num_puestos_max = 2
    ancho_mesa = 60
    for i in range(num_puestos_max):
        x_mesa = x_fisica + (ancho_zona // 2) - (ancho_mesa) + (i * 80)
        y_mesa = y_fisica + 60
        pygame.draw.rect(pantalla, COLOR_MESA, (x_mesa, y_mesa, ancho_mesa, 30), border_radius=3)
        pygame.draw.rect(pantalla, (60, 70, 80), (x_mesa, y_mesa, ancho_mesa, 30), 2, border_radius=3)
        if i < ocupados:
            centro_x = x_mesa + (ancho_mesa // 2)
            centro_y = y_mesa + 15
            pygame.draw.circle(pantalla, COLOR_TRABAJADOR, (centro_x, centro_y), 10)
            pygame.draw.circle(pantalla, (255, 255, 255), (centro_x, centro_y), 10, 2)


indice_datos = 0
animacion_tick = 0
frames_por_dato = 20
ejecutando = True
reloj_generador_cajas = 0

while ejecutando:
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            ejecutando = False

    pantalla.fill(COLOR_FONDO)
    row = df_sub.iloc[indice_datos]


    texto_titulo = fuente_titulo.render("Warehouse Flow and SCADA View (s211 - Urgent First)", True, COLOR_TEXTO)
    pantalla.blit(texto_titulo, (30, 20))
    texto_tiempo = fuente_kpi.render(f"Simulation Time: {row['time_min']:.0f} min", True, (241, 196, 15))
    pantalla.blit(texto_tiempo, (30, 60))


    texto_sec_fisica = fuente_subtitulo.render("PLANT VIEW (PHYSICAL FLOW)", True, (150, 160, 180))
    pantalla.blit(texto_sec_fisica, (ANCHO // 2 - texto_sec_fisica.get_width() // 2, 85))

    dibujar_cinta_transportadora(330, 100, 100, animacion_tick)
    dibujar_cinta_transportadora(710, 100, 100, animacion_tick)

    for caja in todas_las_cajas_fisicas:
        caja.update()
        caja.draw(pantalla)
    todas_las_cajas_fisicas = [c for c in todas_las_cajas_fisicas if c.activa]

    fases_datos_fisicos = [
        ("Picking", row['picking_busy'], zonas_fisicas["Picking"]),
        ("Packing", row['packing_busy'], zonas_fisicas["Packing"]),
        ("Dispatch", row['dispatch_busy'], zonas_fisicas["Dispatch"])
    ]

    for nombre, workers, (x_f, y_f, w_f, h_f) in fases_datos_fisicos:
        pygame.draw.rect(pantalla, COLOR_ZONA_FISICA, (x_f, y_f, w_f, h_f), border_radius=8)
        pygame.draw.rect(pantalla, (80, 90, 105), (x_f, y_f, w_f, h_f), 2, border_radius=8)
        titulo_zona = fuente_titulo.render(nombre, True, COLOR_TEXTO)
        pantalla.blit(titulo_zona, (x_f + (w_f // 2) - (titulo_zona.get_width() // 2), y_f + 10))
        dibujar_puesto_trabajo(x_f, y_f, w_f, int(workers))


    texto_sec_scada = fuente_subtitulo.render("SCADA DASHBOARD (LOGICAL QUEUE STATUS)", True, (150, 160, 180)) # TRADUCIDO
    pygame.draw.line(pantalla, (60, 70, 85), (30, 370), (ANCHO - 30, 370), 3)
    pantalla.blit(texto_sec_scada, (ANCHO // 2 - texto_sec_scada.get_width() // 2, 380))

    fases_datos_scada = [
        ("Picking", row['pick_queue_normal'], row['pick_queue_urgent'], paneles_scada["Picking"]),
        ("Packing", row['pack_queue_normal'], row['pack_queue_urgent'], paneles_scada["Packing"]),
        ("Dispatch", row['dispatch_queue_normal'], row['dispatch_queue_urgent'], paneles_scada["Dispatch"])
    ]

    for nombre, normales, urgentes, (x_p, y_p, w_p, h_p) in fases_datos_scada:
        pygame.draw.rect(pantalla, COLOR_ZONA_FISICA, (x_p, y_p, w_p, h_p), border_radius=8)
        pygame.draw.rect(pantalla, (80, 90, 105), (x_p, y_p, w_p, h_p), 2, border_radius=8)
        titulo_zona_p = fuente_titulo.render(nombre, True, (180, 190, 210))
        pantalla.blit(titulo_zona_p, (x_p + (w_p // 2) - (titulo_zona_p.get_width() // 2), y_p + 15))
        dibujar_silos_capacidad(x_p, y_p, normales, urgentes)


    pygame.display.flip()
    animacion_tick += 2
    reloj_generador_cajas += 1

    if reloj_generador_cajas >= 20:
        todas_las_cajas_fisicas.append(CajaMovil(x_inicio=330, y_inicio=205, target_x=430))
        todas_las_cajas_fisicas.append(CajaMovil(x_inicio=710, y_inicio=205, target_x=810))
        reloj_generador_cajas = 0

    if animacion_tick % (frames_por_dato * 2) == 0:
        indice_datos += 1
        if indice_datos >= len(df_sub):
            indice_datos = 0

    reloj.tick(60)

pygame.quit()
sys.exit()