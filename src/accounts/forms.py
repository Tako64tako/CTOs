# UserCreationFormクラスをインポート
from django.contrib.auth.forms import UserCreationForm, UsernameField
# models.pyで定義したカスタムUserモデルをインポート
from .models import CustomUser
from django import forms

class CustomUserCreationForm(UserCreationForm):
    '''UserCreationFormのサブクラス
    '''
    #username = forms.CharField(label='username', help_text='この項目は必須です。半角アルファベット、半角数字、@/./+/-/_ で150文字以下にしてください。')
    
    class Meta:
        '''UserCreationFormのインナークラス
        
        Attributes:
          model:連携するUserモデル
          fields:フォームで使用するフィールド
        '''
        # 連携するUserモデルを設定
        model = CustomUser
        # フォームで使用するフィールドを設定
        # ユーザー名、メールアドレス、パスワード、パスワード(確認用)
        fields = ('username', 'email', 'password1', 'password2')