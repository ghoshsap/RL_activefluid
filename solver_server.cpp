#include "msgpack.hpp"
#include <vector>
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <string>
#include <stdexcept>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <ostream>
#include <chrono>
#include <ctime>
#include "../cuPSS/inc/cupss.h"
#include <chrono>

#ifdef WITHCUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define NX 64
#define NY 64

int main() {

    int gpu = 1;
    float dx = 1.0f;
    float dy = 1.0f;
    float dt = 0.05f;
    int nn = 1000000000;
    int total_steps = 1000000000;
    int steps_per_l = 5;
    float a2 = -1.0f;
    float a4 = 1.0f;
    float kQ = 4.0f;
    float lambda = 1.0f;
    float gamma = 1.0f;
    float eta = 1.0f;
    float fric = 0.01f;
    float beta = 1.0f;

    float angle_radians = M_PI / 3.0; 
    float nx_ = cos(angle_radians);
    float ny_ = sin(angle_radians);

    evolver system(gpu, NX, NY, dx, dy, dt, 1000000);

    system.createField("Qxx", true);
    system.createField("Qxy", true);
    system.createField("alpha", false);
    system.createField("iqxQxx", false);
    system.createField("iqyQxx", false);
    system.createField("iqxQxy", false);
    system.createField("iqyQxy", false);
    system.createField("sigxx", false);
    system.createField("sigxy", false);
    system.createField("vx", false);
    system.createField("vy", false);
    system.createField("w", false);
    system.createField("Q2", false);

    system.addParameter("a2", a2);
    system.addParameter("a4", a4);
    system.addParameter("kQ", kQ);
    system.addParameter("lambda", lambda);
    system.addParameter("gamma", gamma);
    system.addParameter("eta", eta);
    system.addParameter("fric", fric);
    system.addParameter("beta", beta);

    system.addEquation("dt Qxx + (a2 + kQ*q^2)*Qxx = -a4*Q2*Qxx - vx*iqxQxx - vy*iqyQxx + lambda*iqx*vx - 2*Qxy*w");
    system.addEquation("dt Qxy + (a2 + kQ*q^2)*Qxy = -a4*Q2*Qxy - vx*iqxQxy - vy*iqyQxy + 0.5*lambda*iqx*vy + 0.5*lambda*iqy*vx + 2*Qxx*w");
    system.addEquation("iqxQxx = iqx*Qxx");
    system.addEquation("iqxQxy = iqx*Qxy");
    system.addEquation("iqyQxx = iqy*Qxx");
    system.addEquation("iqyQxy = iqy*Qxy");

    system.addEquation("sigxx = beta * alpha * Qxx");
    system.addEquation("sigxy = beta * alpha * Qxy");
    system.addEquation("w = 0.5*iqx*vy-0.5*iqy*vx");
    system.addEquation("Q2 = Qxx^2 + Qxy^2");

    system.addEquation("vx * (fric + eta*q^2) = (iqx + iqx^3*1/q^2 - iqx*iqy^2*1/q^2) * sigxx + (iqy + iqx^2* iqy*1/q^2 + iqx^2*iqy*1/q^2) * sigxy");
    system.addEquation("vy * (fric + eta*q^2) = (iqx + iqx*iqy^2*1/q^2 + iqx*iqy^2*1/q^2) * sigxy + (-iqy - iqy^3*1/q^2 + iqx^2*iqy*1/q^2) * sigxx");
   
    //system.addNoise("Qxx", "0.5*q^2");
    //system.addNoise("Qxy", "0.5*q^2");

 
    system.printInformation();

    std::srand(1324);
    for (int j = 0; j < NY; j++)
    {
        for (int i = 0; i < NX; i++)
        {
            int index = j * NX + i;
            system.fields[0]->real_array[index].x = (nx_*nx_ - 0.5) + 0.001f * (float)(rand()%200-100);
            system.fields[1]->real_array[index].x = nx_*ny_ + 0.001f * (float)(rand()%200-100);
        }
    }

    system.prepareProblem();


    for (int i = 0; i < system.fields.size(); i++)
    {
        system.fields[i]->outputToFile = false;
    }

    // system.setOutputField("Q2", true);
    // system.setOutputField("alpha", true);

    // system.setOutputField("vx", true);
    // system.setOutputField("vy", true);

    // Create a socket
    int server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        std::cerr << "Error creating socket\n";
        return 1;
    }

    // Bind the socket to a port
    struct sockaddr_in server_address;
    server_address.sin_family = AF_INET;
    server_address.sin_addr.s_addr = INADDR_ANY;
    server_address.sin_port = htons(8090); 

    if (bind(server_socket, (struct sockaddr *)&server_address, sizeof(server_address)) < 0) {
        std::cerr << "Error binding socket\n";
        return 1;
    }

    listen(server_socket, 1);

    int global_steps = 0;

    float2 alpha_temp[NX*NY];
    float2 Qxx_temp[NX*NY];
    float2 Qxy_temp[NX*NY];
    float2 vx_temp[NX*NY];
    float2 vy_temp[NX*NY];
    for (int i = 0; i < NX*NY; i++)
    {        
        alpha_temp[i].x = 0.0f;
        alpha_temp[i].y = 0.0f;
    }


    float data_to_send[4*NX*NY]; 
    
    int frame_count = 0;
    auto start = std::chrono::system_clock::now();

    // int client_socket = accept(server_socket, nullptr, nullptr);

    while (true) {

        // ##################################
        // RECEIVE ALPHA FIELD THROUGH SOCKET
        // ##################################

        // Accept incoming connection
        int client_socket = accept(server_socket, nullptr, nullptr);
        if (client_socket < 0) {

            sleep(1);
            std::cerr << "Error accepting connection\n";
            return 1;
        }

        int x_bytes = 36867;
        char buffer[36867];

        // Receive data from the client
        std::string received_data = "";

        while (received_data.length() < x_bytes) {
            char chunk[x_bytes - received_data.length()];
            int bytes_received = recv(client_socket, chunk, x_bytes - received_data.length(), 0);

            if (bytes_received <= 0) {
                std::cerr << "Error receiving data\n";
                close(client_socket);
                return 1;
            }
            received_data += std::string(chunk, bytes_received);
        }


        // Copy received data to buffer before unpacking
        std::copy(received_data.begin(), received_data.end(), buffer);

        // #############################
        //     CONVERTING TO FLOAT2
        // #############################

        // Create a vector to store received numbers
        std::vector<float> received_numbers;
        // Deserialize the received data
        msgpack::unpacked msg;
        msgpack::unpack(msg, buffer, received_data.size());
        msg.get().convert(received_numbers);

        // std::cout << "recvd data ->" << global_steps << std::endl;

        int index = 0;
        for (int i = 0; i < NX; i++)
        {
            for (int j = 0; j < NY; j++)
            {
                index = j*NX + i;
                alpha_temp[index].x = received_numbers[index];
                // printf("(%d, %d) --> %f\n", i, j, received_numbers[index]);
                // std::cout << (i, j) << alpha_real[i][j] << " ";
            }
        }


        float sum = 0;
        for (const auto& num : received_numbers) {
            sum += num;            

        }


        // ###########################
        //      RESET CHECK
        // ###########################

        if (sum > 4095)
        {
            // std::cout << global_steps << "resetting" << std::endl;

            srand(static_cast<unsigned int>(time(nullptr)));
            float random_fraction = 0.1666f + static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * (0.5f - 0.1666f);
            //float random_fraction = 0.25f ;
            // random_fraction = 0.33333;
            float angle_radians = random_fraction*M_PI; 
            nx_ = cos(angle_radians);
            ny_ = sin(angle_radians);

            // std::cout << random_fraction << std::endl;
            // std::cout << nx_ << std::endl;
            // std::cout << ny_ << std::endl;

            for (int j = 0; j < NY; j++)
            {
                for (int i = 0; i < NX; i++)
                {
                    int index = j * NX + i;
                    system.fields[0]->real_array[index].x =  (nx_*nx_ - 0.5) + 0.001f * (float)(rand()%200-100);
                    system.fields[1]->real_array[index].x =   nx_*ny_  + 0.001f * (float)(rand()%200-100);
                }
            }
            cudaMemcpy(system.fields[0]->real_array_d, system.fields[0]->real_array, NX*NY*sizeof(float2), cudaMemcpyHostToDevice);
            system.fields[0]->toComp();
            cudaMemcpy(system.fields[1]->real_array_d, system.fields[1]->real_array, NX*NY*sizeof(float2), cudaMemcpyHostToDevice);
            system.fields[1]->toComp();
        }


        // ###########################
        //      COPY DATA TO GPU
        // ###########################

        // cudaMemcpy(destination pointer, origin pointer, size, direction);
        cudaMemcpy(system.fields[2]->real_array_d, alpha_temp, NX*NY*sizeof(float2), cudaMemcpyHostToDevice);
        // cudaMemcpy(system.fields[2]->real_array, alpha_temp, NX*NY*sizeof(float2), cudaMemcpyHostToHost);
        system.fields[2]->toComp();

        // ###########################
        //         RUN SOLVER
        // ###########################
        for (int tt = 0; tt < steps_per_l; tt++)
        {
            system.advanceTime();
            // std::cout << "taken one time step" << std::endl;
            frame_count++;
        }
        // if (frame_count > 1000)
        // {
        //     auto now = std::chrono::system_clock::now();
        //     std::chrono::duration<double> elapsed_seconds = now - start;
        //     double frame_rate = (double)frame_count / elapsed_seconds.count();
        //     std::cout << "Frame rate (fps) " << frame_rate << "\r" << std::endl;
        //     std::cout.flush();
        //     frame_count = 0;
        //     auto start = std::chrono::system_clock::now();
        // }

        // ###########################
        //         COPY TO RAM
        // ###########################
        cudaMemcpy(Qxx_temp, system.fields[0]->real_array_d, NX*NY*sizeof(float2), cudaMemcpyDeviceToHost);
        cudaMemcpy(Qxy_temp, system.fields[1]->real_array_d, NX*NY*sizeof(float2), cudaMemcpyDeviceToHost);
        cudaMemcpy(vx_temp, system.fields[9]->real_array_d, NX*NY*sizeof(float2), cudaMemcpyDeviceToHost);
        cudaMemcpy(vy_temp, system.fields[10]->real_array_d, NX*NY*sizeof(float2), cudaMemcpyDeviceToHost);

        // #####################
        // CONCATENATE DATA
        // #####################
        int count  = 0;
        for (int kk = 0; kk < NX*NY; kk++)
        {
            data_to_send[count] = Qxx_temp[kk].x;
            count = count + 1;
        }
        for (int kk = 0; kk < NX*NY; kk++)
        {
            data_to_send[count] = Qxy_temp[kk].x;
            count  = count + 1;
        }
        for (int kk = 0; kk < NX*NY; kk++)
        {
            data_to_send[count] = vx_temp[kk].x;
            count = count + 1;
        }
        for (int kk = 0; kk < NX*NY; kk++)
        {
            data_to_send[count] = vy_temp[kk].x;
            count = count + 1;
        }

        // #####################
        // SEND DATA BACK
        // #####################

        // Pack data
        std::stringstream packed_data_stream;
        msgpack::pack(packed_data_stream, data_to_send);
        std::string packed_data = packed_data_stream.str();

        // sleep(0.5);
        // Send data
        // while (!packed_data.empty()) {

        //     // std::cout << packed_data.size() << std::endl;

        //     ssize_t sent = send(client_socket, packed_data.data(), packed_data.size(), 0);
        //     if (sent < 0) {
        //         std::cerr << "Error sending data\n";
        //         return 1;
        //     }

        //     packed_data = packed_data.substr(sent);
        // }

        ssize_t total_sent = 0;
        while (total_sent < packed_data.size()) {
            ssize_t sent = send(client_socket, packed_data.data() + total_sent, packed_data.size() - total_sent, 0);
            if (sent < 0) {
                std::cerr << "Error sending data\n";
                return 1;
            }
            total_sent += sent;
        }

        // Close client socket
        close(client_socket);

        // std::cout << "sent data ->" << global_steps << std::endl;

        global_steps++;
    }

    return 0;
}
