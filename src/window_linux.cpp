#include "window_linux.h"
#include <iostream>

MyWindow::MyWindow(int width, int height, int offset_w, int offset_h)
    : m_width(width), m_height(height), m_offset_w(offset_w), m_offset_h(offset_h)
{
    m_display = XOpenDisplay(NULL);
    if (!m_display) {
        std::cerr << "Cannot open display" << std::endl;
        exit(1);
    }

    m_screen = DefaultScreen(m_display);

    m_window = XCreateSimpleWindow(
        m_display,
        RootWindow(m_display, m_screen),
        m_offset_w, m_offset_h,
        m_width, m_height,
        1,
        BlackPixel(m_display, m_screen),
        BlackPixel(m_display, m_screen)
    );

    // Select input events
    XSelectInput(m_display, m_window, ExposureMask | KeyPressMask | ButtonPressMask | StructureNotifyMask);

    // Handle window close
    m_wmDelete = XInternAtom(m_display, "WM_DELETE_WINDOW", True);
    XSetWMProtocols(m_display, m_window, &m_wmDelete, 1);

    // Create graphics context
    m_gc = XCreateGC(m_display, m_window, 0, NULL);
    XSetForeground(m_display, m_gc, WhitePixel(m_display, m_screen));

    // Map the window
    XMapWindow(m_display, m_window);
    XFlush(m_display);
}

MyWindow::~MyWindow()
{
    XFreeGC(m_display, m_gc);
    XDestroyWindow(m_display, m_window);
    XCloseDisplay(m_display);
}

bool MyWindow::ProcessMessages()
{
    XEvent event;
    while (XPending(m_display)) {
        XNextEvent(m_display, &event);
        switch (event.type) {
        case ClientMessage:
            if (event.xclient.data.l[0] == (long)m_wmDelete) {
                return false;  // Quit on window close
            }
            break;
        case DestroyNotify:
            return false;
        }
    }
    return true;
}

void MyWindow::DrawPixels(std::vector<std::vector<int>> pixels)
{
    // Clear the window
    XClearWindow(m_display, m_window);

    // Draw pixels
    for (const auto& pixel : pixels) {
        if (pixel.size() >= 2) {
            XDrawPoint(m_display, m_window, m_gc, pixel[0], pixel[1]);
        }
    }

    XFlush(m_display);
}
