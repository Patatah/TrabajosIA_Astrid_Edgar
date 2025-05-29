from Trainer import EmotionResNetTrainer
from multiprocessing import freeze_support


def main():
    # 1. Inicializar el entrenador
    trainer = EmotionResNetTrainer(
        train_dir='C:/Users/Propietario/Documents/Edgar/IA/FANE/fane_split/train',
        val_dir='C:/Users/Propietario/Documents/Edgar/IA/FANE/fane_split/val',
        num_classes=5,
        batch_size=32
    )

    # 2. Configurar pesos para clases desbalanceadas
    class_counts = [1419, 2320, 1536, 2185, 1483] 
    trainer.set_class_weights(class_counts)

    # 3. Entrenar el modelo
    history = trainer.train(num_epochs=25)

    # 4. Evaluar el modelo
    val_loss, val_acc = trainer.evaluate()

    # 5. Guardar el modelo
    trainer.save_model('emotion_resnet50.pth')

    # 6. Visualizar resultados
    trainer.plot_confusion_matrix()
    trainer.visualize_predictions()

if __name__ == '__main__':
    freeze_support()  # Necesario para Windows
    main()
