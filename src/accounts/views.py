from django.shortcuts import render
from django.views.generic import CreateView, TemplateView
from django.urls import reverse_lazy
from .forms import CustomUserCreationForm
# from django.contrib.auth.mixins import LoginRequiredMixin
# from django.contrib.auth.views import PasswordChangeView,\
#                                       PasswordChangeDoneView,\
#                                       PasswordResetView,\
#                                       PasswordResetDoneView,\
#                                       PasswordResetConfirmView,\
#                                       PasswordResetCompleteView

class SignUpView(CreateView):
    '''サインアップページのビュー
    
    '''
    # forms.pyで定義したフォームのクラス
    form_class = CustomUserCreationForm
    # レンダリングするテンプレート
    template_name = "signup.html"
    # サインアップ完了後のリダイレクト先のURLパターン
    success_url = reverse_lazy('accounts:signup_success')

    def form_valid(self, form):
        '''CreateViewクラスのform_valid()をオーバーライド
        
        フォームのバリデーションを通過したときに呼ばれる
        フォームデータの登録を行う
        
        parameters:
          form(django.forms.Form):
            form_classに格納されているCustomUserCreationFormオブジェクト
        Return:
          HttpResponseRedirectオブジェクト:
            スーパークラスのform_valid()の戻り値を返すことで、
            success_urlで設定されているURLにリダイレクトさせる
        '''
        # formオブジェクトのフィールドの値をデータベースに保存
        user = form.save()
        self.object = user
        # 戻り値はスーパークラスのform_valid()の戻り値(HttpResponseRedirect)
        return super().form_valid(form)

class SignUpSuccessView(TemplateView):
    '''サインアップ成功ページのビュー
    
    '''
    # レンダリングするテンプレート
    template_name = "signup_success.html"
    
# class SuccessLoginView(TemplateView):
#     '''ログイン成功ページのビュー
    
#     '''
#     # レンダリングするテンプレート
#     template_name = "index.html"

# # --- パスワード変更と変更完了
# class PasswordChange(LoginRequiredMixin, PasswordChangeView):
#     '''パスワード変更ビュー
    
#     '''
#     success_url = reverse_lazy('accounts:password_change_done')
#     template_name = 'password_change.html'

#     def get_context_data(self, **kwargs):
#         context = super().get_context_data(**kwargs) # 継承元のメソッドCALL
#         context["form_name"] = "password_change"
#         return context

# class PasswordChangeDone(LoginRequiredMixin,PasswordChangeDoneView):
#     '''パスワード変更完了
    
#     '''
#     template_name = 'password_change_done.html'

# # --- ここから追加
# class PasswordReset(PasswordResetView):
#     """パスワード変更用URLの送付ページ"""
#     #subject_template_name = 'accounts/mail_template/reset/subject.txt'
#     #email_template_name = 'accounts/mail_template/reset/message.txt'
#     template_name = 'password_reset_form.html'
#     success_url = reverse_lazy('password_reset_done')


# class PasswordResetDone(PasswordResetDoneView):
#     """パスワード変更用URLを送りましたページ"""
#     template_name = 'accounts/password_reset_done.html'


# class PasswordResetConfirm(PasswordResetConfirmView):
#     """新パスワード入力ページ"""
#     success_url = reverse_lazy('accounts:password_reset_complete')
#     template_name = 'password_reset_confirm.html'


# class PasswordResetComplete(PasswordResetCompleteView):
#     """新パスワード設定しましたページ"""
#     template_name = 'password_reset_complete.html'

# # --- ここまで