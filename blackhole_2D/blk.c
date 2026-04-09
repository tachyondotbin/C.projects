#include <SDL2/SDL.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>

#define WIDTH  1920
#define HEIGHT 1080
#define N_PARTICLES 150000
#define BH_RADIUS 1

typedef struct {
    float x, y, vx, vy;
    char type;
} Particle;

Particle particles[N_PARTICLES];
SDL_Window* window = NULL;
SDL_Renderer* renderer = NULL;
float bh_x = WIDTH/2.0f, bh_y = HEIGHT/2.0f;
int running = 1;

void init_particles() {
    srand(5);
    for (int i = 0; i < N_PARTICLES/2; i++) {
        particles[i].x = 200;
        particles[i].y = 200 + rand()%700;
        particles[i].vx = 0.2f;
        particles[i].vy = (rand()%400-200)*0.002f;
        particles[i].type = 0;  // Красный газ
    }
    for (int i = N_PARTICLES/2; i < N_PARTICLES; i++) {
        particles[i].x = WIDTH-350;
        particles[i].y = 400 + rand()%400;
        particles[i].vx = -0.2f;
        particles[i].vy = (rand()%400-200)*0.002f;
        particles[i].type = 1;  // Синий газ
    }
}

void update_physics_range(int start, int end) {
    for (int i = start; i < end; i++) {
        Particle* p = &particles[i];
        float dx = bh_x - p->x;
        float dy = bh_y - p->y;
        float r = sqrtf(dx*dx + dy*dy) + 1.0f;
        float force = 400000.0f / (r*r);
        p->vx += (dx/r) * force * 0.00015f;
        p->vy += (dy/r) * force * 0.00015f;
        p->x += p->vx;
        p->y += p->vy;

        
        if (r < BH_RADIUS) {
            float angle = ((float)rand() / RAND_MAX) * 2 * M_PI;
            float radius = ((float)rand() / RAND_MAX) * BH_RADIUS;
            p->x = bh_x + radius * cosf(angle);
            p->y = bh_y + radius * sinf(angle);
            p->vx = (rand()%200-100)*0.008f;
            p->vy = (rand()%200-100)*0.008f;
        }
    }
}


void render() {
    SDL_SetRenderDrawColor(renderer, 5, 5, 15, 255);
    SDL_RenderClear(renderer);

    // дыра
    for (int r = 0; r < BH_RADIUS; r++) {
        SDL_SetRenderDrawColor(renderer, 30-r, 10, 50-r, 255);
        for (int theta = 0; theta < 360; theta += 8) {
            float rad = theta * M_PI / 180.0f;
            int x = (int)(bh_x + r * cosf(rad));
            int y = (int)(bh_y + r * sinf(rad));
            if (x > 0 && x < WIDTH && y > 0 && y < HEIGHT)
                SDL_RenderDrawPoint(renderer, x, y);
        }
    }

    // Частицы
    for (int i = 0; i < N_PARTICLES/2; i++) {
        int x = (int)particles[i].x;
        int y = (int)particles[i].y;
        if (x > 0 && x < WIDTH && y > 0 && y < HEIGHT) {
            SDL_SetRenderDrawColor(renderer, 255, 80, 80, 255);  // Красный
            SDL_RenderDrawPoint(renderer, x, y);
        }
    }
    for (int i = N_PARTICLES/2; i < N_PARTICLES; i++) {
        int x = (int)particles[i].x;
        int y = (int)particles[i].y;
        if (x > 0 && x < WIDTH && y > 0 && y < HEIGHT) {
            SDL_SetRenderDrawColor(renderer, 80, 80, 255, 255);  // Синий
            SDL_RenderDrawPoint(renderer, x, y);
        }
    }

    SDL_RenderPresent(renderer);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    init_particles();

    if (rank == 0) {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            printf("SDL error: %s\n", SDL_GetError());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        window = SDL_CreateWindow(
            "",
            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
            WIDTH, HEIGHT, SDL_WINDOW_SHOWN
        );

        if (!window) {
            printf("Window error: %s\n", SDL_GetError());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        if (!renderer) {
            printf("Renderer error: %s\n", SDL_GetError());
            SDL_DestroyWindow(window);
            SDL_Quit();
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    SDL_Event event;
    while (1) {
        int chunk = N_PARTICLES / size;
        int start = rank * chunk;
        int end = (rank == size-1) ? N_PARTICLES : start + chunk;
        update_physics_range(start, end);


        MPI_Gather(particles + start, (end-start)*sizeof(Particle), MPI_BYTE,
                   particles, (end-start)*sizeof(Particle), MPI_BYTE,
                   0, MPI_COMM_WORLD);

        if (rank == 0) {
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT || 
                    (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE))
                    running = 0;
            }

            render();

            SDL_Delay(16);
            if (!running) break;
        }
    }

    if (rank == 0) {
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }

    MPI_Finalize();
    return 0;
}
