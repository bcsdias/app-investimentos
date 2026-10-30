import React, { useState } from 'react';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { Modal } from '../components/common/Modal';

describe('Modal Acessibilidade (A11y)', () => {
  it('Modal é rotulado pelo título via aria-labelledby', () => {
    render(
      <Modal isOpen={true} onClose={vi.fn()} title="Confirmar Operação">
        <p>Conteúdo explicativo</p>
      </Modal>
    );

    const dialog = screen.getByRole('dialog', { name: 'Confirmar Operação' });
    expect(dialog).toBeInTheDocument();

    const heading = screen.getByRole('heading', { level: 2, name: 'Confirmar Operação' });
    const headingId = heading.getAttribute('id');
    expect(headingId).toBeTruthy();
    expect(dialog).toHaveAttribute('aria-labelledby', headingId);
  });

  it('foco inicial vai para o primeiro elemento focável dentro do modal ao abrir', () => {
    function Wrapper() {
      const [isOpen, setIsOpen] = useState(false);
      return (
        <div>
          <button id="open-btn" onClick={() => setIsOpen(true)}>
            Abrir Modal
          </button>
          <Modal isOpen={isOpen} onClose={() => setIsOpen(false)} title="Modal com Campos">
            <input aria-label="Nome do titular" id="name-input" />
            <button id="submit-btn">Salvar</button>
          </Modal>
        </div>
      );
    }

    render(<Wrapper />);
    const openBtn = screen.getByRole('button', { name: 'Abrir Modal' });
    fireEvent.click(openBtn);

    // O primeiro elemento focável no Modal com onClose é o botão de fechar (X), ou o primeiro input se não houver close
    const closeBtn = screen.getByRole('button', { name: 'Fechar modal' });
    expect(document.activeElement).toBe(closeBtn);
  });

  it('Modal prende o foco com Tab e Shift+Tab (focus trap)', () => {
    render(
      <Modal isOpen={true} onClose={vi.fn()} title="Modal Foco">
        <input aria-label="Campo 1" id="field-1" />
        <button id="action-btn">Confirmar</button>
      </Modal>
    );

    const closeBtn = screen.getByRole('button', { name: 'Fechar modal' });
    const actionBtn = screen.getByRole('button', { name: 'Confirmar' });

    // Coloca o foco no último elemento (actionBtn)
    actionBtn.focus();
    expect(document.activeElement).toBe(actionBtn);

    // Tab no último elemento deve redirecionar o foco para o primeiro (closeBtn)
    fireEvent.keyDown(window, { key: 'Tab' });
    expect(document.activeElement).toBe(closeBtn);

    // Shift+Tab no primeiro elemento (closeBtn) deve redirecionar o foco para o último (actionBtn)
    fireEvent.keyDown(window, { key: 'Tab', shiftKey: true });
    expect(document.activeElement).toBe(actionBtn);
  });

  it('Modal devolve o foco ao elemento que abriu ao fechar', () => {
    function Wrapper() {
      const [isOpen, setIsOpen] = useState(false);
      return (
        <div>
          <button id="trigger-btn" onClick={() => setIsOpen(true)}>
            Abrir Diálogo
          </button>
          <Modal isOpen={isOpen} onClose={() => setIsOpen(false)} title="Diálogo">
            <p>Texto</p>
          </Modal>
        </div>
      );
    }

    render(<Wrapper />);
    const triggerBtn = screen.getByRole('button', { name: 'Abrir Diálogo' });
    triggerBtn.focus();
    expect(document.activeElement).toBe(triggerBtn);

    fireEvent.click(triggerBtn);
    expect(screen.getByRole('dialog')).toBeInTheDocument();

    const closeBtn = screen.getByRole('button', { name: 'Fechar modal' });
    fireEvent.click(closeBtn);

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    expect(document.activeElement).toBe(triggerBtn);
  });

  it('clique no backdrop chama onClose, mas clique no conteúdo interno não fecha', () => {
    const handleClose = vi.fn();
    render(
      <Modal isOpen={true} onClose={handleClose} title="Teste Backdrop">
        <div data-testid="modal-content">
          <p>Conteúdo interno</p>
        </div>
      </Modal>
    );

    const content = screen.getByTestId('modal-content');
    fireEvent.click(content);
    expect(handleClose).not.toHaveBeenCalled();

    const dialogBackdrop = screen.getByRole('dialog');
    fireEvent.click(dialogBackdrop);
    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it('tecla Escape fecha o modal', () => {
    const handleClose = vi.fn();
    render(
      <Modal isOpen={true} onClose={handleClose} title="Teste Escape">
        <p>Conteúdo</p>
      </Modal>
    );

    fireEvent.keyDown(window, { key: 'Escape' });
    expect(handleClose).toHaveBeenCalledTimes(1);
  });
});
