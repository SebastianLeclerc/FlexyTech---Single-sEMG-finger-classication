#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>

#define BUFLEN 1024

int main() {
    // Create socket
    int sockfd;
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }

    // Set up server address and port
    struct sockaddr_in servaddr;
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = inet_addr("192.168.0.4"); // Replace with Beaglebone Green's IP address
    servaddr.sin_port = htons(8080);

    // Connect socket to server address
    if (connect(sockfd, (const struct sockaddr*)&servaddr, sizeof(servaddr)) < 0) {
        perror("connect failed");
        exit(EXIT_FAILURE);
    }

    char buf[BUFLEN];
    int len;

    while (1) {
        // Receive data from server
        if ((len = recv(sockfd, buf, BUFLEN, 0)) < 0) {
            perror("recv failed");
            exit(EXIT_FAILURE);
        }
        buf[len] = '\0'; // Add null terminator to string

        printf("%s\n", buf); // Print received data
    }

    close(sockfd);

    return 0;
}
