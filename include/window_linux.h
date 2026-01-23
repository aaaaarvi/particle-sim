#pragma once

#include <X11/Xlib.h>
#include <vector>

class MyWindow {
public:
    MyWindow(int width, int height, int offset_w, int offset_h);
    MyWindow(int width, int height) : MyWindow(width, height, 100, 100) {};
    MyWindow() : MyWindow(720, 720, 100, 100) {};
    MyWindow(const MyWindow&) = delete;
    MyWindow& operator =(const MyWindow&) = delete;
    ~MyWindow();

    bool ProcessMessages();
    void DrawPixels(std::vector<std::vector<int>> pixels);

private:
    int m_width;
    int m_height;
    int m_offset_w;
    int m_offset_h;
    Display* m_display;
    ::Window m_window;
    int m_screen;
    GC m_gc;
    Atom m_wmDelete;
};
