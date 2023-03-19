#pragma once

#include <Windows.h>
#include "Vector.h"

LRESULT CALLBACK WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

class Window
{
public:
    Window(int width, int height, int offset_w, int offset_h);
    Window(int width, int height) : Window(width, height, 100, 100) {};
    Window() : Window(720, 720, 100, 100) {};
    Window(const Window&) = delete;
    Window& operator =(const Window&) = delete;
    ~Window();

    bool ProcessMessages();
    void DrawPixels(std::vector<std::vector<int>> pixels);

private:
    int m_width;
    int m_height;
    int m_offset_w;
    int m_offset_h;
    HINSTANCE m_hInstance;
    HWND m_hWnd;
};
