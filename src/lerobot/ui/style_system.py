#!/usr/bin/env python

"""
Tailwind-like styling utility for PySide6/Qt applications.
Provides consistent, reusable style classes similar to Tailwind CSS.
"""

from typing import Optional
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt


class StyleSystem:
    """Utility class for applying consistent styles to Qt widgets."""
    
    # Color palette (similar to Tailwind)
    COLORS = {
        # Grays
        'gray-50': '#f8f9fa',
        'gray-100': '#e9ecef',
        'gray-200': '#dee2e6',
        'gray-300': '#ced4da',
        'gray-400': '#adb5bd',
        'gray-500': '#6c757d',
        'gray-600': '#495057',
        'gray-700': '#343a40',
        'gray-800': '#2c3e50',
        'gray-900': '#1e2329',
        
        # Blues
        'blue-400': '#60a5fa',
        'blue-500': '#3498db',
        'blue-600': '#2980b9',
        'blue-700': '#21618c',
        
        # Greens
        'green-400': '#4ade80',
        'green-500': '#27ae60',
        'green-600': '#229954',
        'green-700': '#1e8449',
        
        # Reds
        'red-400': '#f87171',
        'red-500': '#e74c3c',
        'red-600': '#c0392b',
        'red-700': '#a93226',
        
        # Yellows/Oranges
        'yellow-400': '#fbbf24',
        'yellow-500': '#f39c12',
        'yellow-600': '#d97706',
        
        'orange-400': '#fb923c',
        'orange-500': '#e67e22',
        
        # Purples
        'purple-400': '#a78bfa',
        'purple-500': '#9b59b6',
        'purple-600': '#8e44ad',
        'purple-700': '#7d3c98',
        
        # Base colors
        'white': '#ffffff',
        'black': '#000000',
    }
    
    @staticmethod
    def button(
        bg_color: str = 'blue-500',
        text_color: str = 'white',
        hover_color: Optional[str] = None,
        pressed_color: Optional[str] = None,
        size: str = 'md',  # sm, md, lg
        rounded: bool = True
    ) -> str:
        """Generate button style."""
        hover = hover_color or StyleSystem._darker(bg_color)
        pressed = pressed_color or StyleSystem._darker(hover)
        
        sizes = {
            'sm': {'padding': '8px 16px', 'font-size': '10pt'},
            'md': {'padding': '12px 24px', 'font-size': '11pt'},
            'lg': {'padding': '16px 32px', 'font-size': '12pt'},
        }
        size_style = sizes.get(size, sizes['md'])
        
        border_radius = '6px' if rounded else '0px'
        
        return f"""
            QPushButton {{
                background: {StyleSystem.COLORS[bg_color]};
                color: {StyleSystem.COLORS[text_color]};
                font-size: {size_style['font-size']};
                font-weight: 600;
                padding: {size_style['padding']};
                border: 1px solid {StyleSystem._darker(bg_color)};
                border-radius: {border_radius};
            }}
            QPushButton:hover {{
                background: {StyleSystem.COLORS[hover]};
                border: 1px solid {StyleSystem._darker(hover)};
            }}
            QPushButton:pressed {{
                background: {StyleSystem.COLORS[pressed]};
            }}
            QPushButton:disabled {{
                background: {StyleSystem.COLORS['gray-600']};
                color: {StyleSystem.COLORS['gray-400']};
                border: 1px solid {StyleSystem.COLORS['gray-700']};
            }}
        """
    
    @staticmethod
    def card(
        bg_color: str = 'gray-800',
        border_color: str = 'gray-700',
        padding: str = '12px',
        rounded: bool = True
    ) -> str:
        """Generate card/panel style."""
        border_radius = '8px' if rounded else '0px'
        return f"""
            QWidget {{
                background: {StyleSystem.COLORS[bg_color]};
                border: 2px solid {StyleSystem.COLORS[border_color]};
                border-radius: {border_radius};
                padding: {padding};
            }}
        """
    
    @staticmethod
    def input(
        bg_color: str = 'gray-900',
        border_color: str = 'gray-700',
        focus_color: str = 'blue-500',
        text_color: str = 'gray-100'
    ) -> str:
        """Generate input field style."""
        return f"""
            QLineEdit, QPlainTextEdit, QTextEdit {{
                background-color: {StyleSystem.COLORS[bg_color]};
                color: {StyleSystem.COLORS[text_color]};
                border: 2px solid {StyleSystem.COLORS[border_color]};
                border-radius: 4px;
                padding: 8px;
                font-size: 10pt;
            }}
            QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus {{
                border-color: {StyleSystem.COLORS[focus_color]};
            }}
        """
    
    @staticmethod
    def label(
        text_color: str = 'gray-100',
        size: str = 'md',  # sm, md, lg, xl
        weight: str = 'normal'  # normal, medium, semibold, bold
    ) -> str:
        """Generate label style."""
        sizes = {
            'sm': '9pt',
            'md': '10pt',
            'lg': '11pt',
            'xl': '12pt',
        }
        weights = {
            'normal': '400',
            'medium': '500',
            'semibold': '600',
            'bold': '700',
        }
        return f"""
            QLabel {{
                color: {StyleSystem.COLORS[text_color]};
                font-size: {sizes.get(size, '10pt')};
                font-weight: {weights.get(weight, '400')};
            }}
        """
    
    @staticmethod
    def badge(
        bg_color: str = 'blue-500',
        text_color: str = 'white',
        size: str = 'md'
    ) -> str:
        """Generate badge/status indicator style."""
        sizes = {
            'sm': {'padding': '4px 8px', 'font-size': '9pt'},
            'md': {'padding': '6px 12px', 'font-size': '10pt'},
            'lg': {'padding': '8px 16px', 'font-size': '11pt'},
        }
        size_style = sizes.get(size, sizes['md'])
        return f"""
            QLabel {{
                background-color: {StyleSystem.COLORS[bg_color]};
                color: {StyleSystem.COLORS[text_color]};
                padding: {size_style['padding']};
                font-size: {size_style['font-size']};
                font-weight: 600;
                border-radius: 6px;
                border: 1px solid {StyleSystem._darker(bg_color)};
            }}
        """
    
    @staticmethod
    def group_box(
        title_color: str = 'blue-500',
        bg_color: str = 'gray-800',
        border_color: str = 'gray-700'
    ) -> str:
        """Generate group box style."""
        return f"""
            QGroupBox {{
                font-size: 11pt;
                font-weight: 600;
                color: {StyleSystem.COLORS['gray-100']};
                background: {StyleSystem.COLORS[bg_color]};
                border: 2px solid {StyleSystem.COLORS[border_color]};
                border-radius: 8px;
                padding-top: 22px;
                padding-bottom: 18px;
                padding-left: 18px;
                padding-right: 18px;
                margin-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 18px;
                padding: 0 10px 0 10px;
                color: {StyleSystem.COLORS[title_color]};
            }}
        """
    
    @staticmethod
    def apply(widget: QWidget, *styles: str):
        """Apply multiple style strings to a widget."""
        combined = '\n'.join(styles)
        widget.setStyleSheet(combined)
    
    @staticmethod
    def _darker(color_key: str) -> str:
        """Get a darker shade of a color."""
        color_map = {
            'blue-500': 'blue-600',
            'blue-600': 'blue-700',
            'green-500': 'green-600',
            'green-600': 'green-700',
            'red-500': 'red-600',
            'red-600': 'red-700',
            'purple-500': 'purple-600',
            'purple-600': 'purple-700',
            'yellow-500': 'yellow-600',
        }
        return color_map.get(color_key, 'gray-700')
    
    @staticmethod
    def color(key: str) -> str:
        """Get a color value by key."""
        return StyleSystem.COLORS.get(key, '#000000')


# Convenience functions for common patterns
def btn_primary() -> str:
    """Primary button style."""
    return StyleSystem.button('blue-500', 'white')

def btn_success() -> str:
    """Success button style."""
    return StyleSystem.button('green-500', 'white')

def btn_danger() -> str:
    """Danger button style."""
    return StyleSystem.button('red-500', 'white')

def btn_warning() -> str:
    """Warning button style."""
    return StyleSystem.button('yellow-500', 'white')

def btn_purple() -> str:
    """Purple button style."""
    return StyleSystem.button('purple-500', 'white')

def card_dark() -> str:
    """Dark card style."""
    return StyleSystem.card('gray-800', 'gray-700')

def input_dark() -> str:
    """Dark input style."""
    return StyleSystem.input('gray-900', 'gray-700', 'blue-500')

def badge_success() -> str:
    """Success badge style."""
    return StyleSystem.badge('green-500', 'white')

def badge_warning() -> str:
    """Warning badge style."""
    return StyleSystem.badge('yellow-500', 'white')

def badge_danger() -> str:
    """Danger badge style."""
    return StyleSystem.badge('red-500', 'white')

def badge_info() -> str:
    """Info badge style."""
    return StyleSystem.badge('blue-500', 'white')

